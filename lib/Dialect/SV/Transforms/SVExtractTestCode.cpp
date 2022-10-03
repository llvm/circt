//===- SVExtractTestCode.cpp - SV Simulation Extraction Pass --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass extracts simulation constructs to submodules.  It
// will take simulation operations, write, finish, assert, assume, and cover and
// extract them and the dataflow into them into a separate module.  This module
// is then instantiated in the original module.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWSymCache.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/Debug.h"

#include <set>

#define DEBUG_TYPE "extract-test-code"

using namespace mlir;
using namespace circt;
using namespace sv;

//===----------------------------------------------------------------------===//
// StubExternalModules Helpers
//===----------------------------------------------------------------------===//

// Reimplemented from SliceAnalysis to use a worklist rather than recursion and
// non-insert ordered set.
static void getBackwardSliceSimple(Operation *rootOp,
                                   SetVector<Operation *> &backwardSlice,
                                   std::function<bool(Operation *)> filter) {
  SmallVector<Operation *> worklist;
  worklist.push_back(rootOp);

  while (!worklist.empty()) {
    Operation *op = worklist.back();
    worklist.pop_back();

    if (!op || op->hasTrait<mlir::OpTrait::IsIsolatedFromAbove>())
      continue;

    // Evaluate whether we should keep this def.
    // This is useful in particular to implement scoping; i.e. return the
    // transitive backwardSlice in the current scope.
    if (filter && !filter(op))
      continue;

    for (auto en : llvm::enumerate(op->getOperands())) {
      auto operand = en.value();
      if (auto *definingOp = operand.getDefiningOp()) {
        if (!backwardSlice.contains(definingOp))
          worklist.push_back(definingOp);
      } else if (auto blockArg = operand.dyn_cast<BlockArgument>()) {
        Block *block = blockArg.getOwner();
        Operation *parentOp = block->getParentOp();
        // TODO: determine whether we want to recurse backward into the other
        // blocks of parentOp, which are not technically backward unless they
        // flow into us. For now, just bail.
        assert(parentOp->getNumRegions() == 1 &&
               parentOp->getRegion(0).getBlocks().size() == 1);
        if (!backwardSlice.contains(parentOp))
          worklist.push_back(parentOp);
      } else {
        llvm_unreachable("No definingOp and not a block argument.");
      }
    }

    backwardSlice.insert(op);
  }

  // Don't insert the top level operation, we just queried on it and don't
  // want it in the results.
  backwardSlice.remove(rootOp);
}

// Reimplemented and simplified from SliceAnalysis to use a worklist rather than
// recursion and not consider nested regions. Also accepts a clone set
// representing a backward dataflow slice. As soon as a use is found outside the
// backward slice, return false. Otherwise, the whole forward slice is contained
// in the backward slice, and return true.
static bool getForwardSliceSimple(Operation *rootOp,
                                  SetVector<Operation *> &opsToClone,
                                  SetVector<Operation *> &forwardSlice) {
  SmallPtrSet<Operation *, 32> visited;
  SmallVector<Operation *> worklist;
  worklist.push_back(rootOp);

  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    forwardSlice.insert(op);
    visited.insert(op);
    for (auto *user : op->getUsers()) {
      if (!opsToClone.contains(user))
        return false;
      if (!visited.contains(user))
        worklist.push_back(user);
    }
  }

  return true;
}

// Compute the dataflow for a set of ops.
static void dataflowSlice(SetVector<Operation *> &ops,
                          SetVector<Operation *> &results) {
  for (auto op : ops) {
    getBackwardSliceSimple(op, results, [](Operation *testOp) -> bool {
      return !isa<sv::ReadInOutOp>(testOp) && !isa<hw::InstanceOp>(testOp) &&
             !isa<sv::PAssignOp>(testOp) && !isa<sv::BPAssignOp>(testOp);
    });
  }
}

// Compute the ops defining the blocks a set of ops are in.
static void blockSlice(SetVector<Operation *> &ops,
                       SetVector<Operation *> &blocks) {
  for (auto op : ops) {
    while (!isa<hw::HWModuleOp>(op->getParentOp())) {
      op = op->getParentOp();
      blocks.insert(op);
    }
  }
}

// Aggressively mark operations to be moved to the new module.  This leaves
// maximum flexibility for optimization after removal of the nodes from the
// old module.
static SetVector<Operation *> computeCloneSet(SetVector<Operation *> &roots) {
  SetVector<Operation *> results;
  // Get Dataflow for roots
  dataflowSlice(roots, results);

  // Get Blocks
  SetVector<Operation *> blocks;
  blockSlice(roots, blocks);
  blockSlice(results, blocks);

  // Make sure dataflow to block args (if conds, etc) is included
  dataflowSlice(blocks, results);

  // include the blocks and roots to clone
  results.insert(roots.begin(), roots.end());
  results.insert(blocks.begin(), blocks.end());

  return results;
}

// Find instances that only feed the clone set, and add them if possible. This
// also returns a list of ops that should be erased, which includes such
// instances and their forward dataflow slices.
static void addInstancesToCloneSet(
    SetVector<Value> &inputs, SetVector<Operation *> &opsToClone,
    SmallPtrSetImpl<Operation *> &opsToErase,
    DenseMap<StringAttr, SmallPtrSet<Operation *, 32>> &extractedInstances) {
  // Track inputs to add, which are used by the instance that will be extracted.
  SmallVector<Value> inputsToAdd;

  // Track inputs to remove, which come from instances that will be extracted.
  SmallVector<Value> inputsToRemove;

  // Check each input into the clone set.
  for (auto value : inputs) {
    // Check if the input comes from an instance, and it isn't already added to
    // the clone set.
    auto *definingOp = value.getDefiningOp();
    auto instance = dyn_cast_or_null<hw::InstanceOp>(definingOp);
    if (!instance)
      continue;
    if (opsToClone.contains(instance))
      continue;

    // Compute the instance's forward slice. If it wasn't fully contained in
    // the clone set, move along.
    SetVector<Operation *> forwardSlice;
    if (!getForwardSliceSimple(instance, opsToClone, forwardSlice))
      continue;

    // Add the instance to the clone set and mark the input to be removed from
    // the input set. Add any instance inputs to the input set. Also add the
    // instance to the map of extracted instances by module.
    opsToClone.insert(instance);
    for (auto operand : instance.getOperands())
      inputsToAdd.push_back(operand);
    inputsToRemove.push_back(value);
    extractedInstances[instance.getModuleNameAttr().getAttr()].insert(instance);

    // Mark the instance and its forward dataflow to be erased from the pass.
    // Normally, ops in the clone set are canonicalized away later, but for
    // this case, we have to proactively erase them. The instances must be
    // erased because we can't canonicalize away instances with unused results
    // in general. The forward dataflow must be erased because the instance is
    // being erased, and we can't leave null operands after this pass.
    opsToErase.insert(instance);
    for (auto *forwardOp : forwardSlice)
      opsToErase.insert(forwardOp);
  }

  // Remove any inputs marked for removal.
  for (auto v : inputsToRemove)
    inputs.remove(v);

  // Add any inputs marked for addition.
  for (auto v : inputsToAdd)
    inputs.insert(v);
}

static StringRef getNameForPort(Value val, ArrayAttr modulePorts) {
  if (auto readinout = dyn_cast_or_null<ReadInOutOp>(val.getDefiningOp())) {
    if (auto wire = dyn_cast<WireOp>(readinout.getInput().getDefiningOp()))
      return wire.getName();
    if (auto reg = dyn_cast<RegOp>(readinout.getInput().getDefiningOp()))
      return reg.getName();
  } else if (auto bv = val.dyn_cast<BlockArgument>()) {
    return modulePorts[bv.getArgNumber()].cast<StringAttr>().getValue();
  }
  return "";
}

// Given a set of values, construct a module and bind instance of that module
// that passes those values through.  Returns the new module and the instance
// pointing to it.
static hw::HWModuleOp createModuleForCut(hw::HWModuleOp op,
                                         SetVector<Value> &inputs,
                                         BlockAndValueMapping &cutMap,
                                         StringRef suffix, Attribute path,
                                         Attribute fileName) {
  // Filter duplicates and track duplicate reads of elements so we don't
  // make ports for them
  SmallVector<Value> realInputs;
  DenseMap<Value, Value> dups; // wire,reg,lhs -> read
  DenseMap<Value, SmallVector<Value>>
      realReads; // port mapped read -> dup reads
  for (auto v : inputs) {
    if (auto readinout = dyn_cast_or_null<ReadInOutOp>(v.getDefiningOp())) {
      auto op = readinout.getInput();
      if (dups.count(op)) {
        realReads[dups[op]].push_back(v);
        continue;
      }
      dups[op] = v;
    }
    realInputs.push_back(v);
  }

  // Create the extracted module right next to the original one.
  OpBuilder b(op);

  // Construct the ports, this is just the input Values
  SmallVector<hw::PortInfo> ports;
  {
    auto srcPorts = op.getArgNames();
    for (auto &port : llvm::enumerate(realInputs)) {
      auto name = getNameForPort(port.value(), srcPorts);
      ports.push_back({b.getStringAttr(name), hw::PortDirection::INPUT,
                       port.value().getType(), port.index()});
    }
  }

  // Create the module, setting the output path if indicated.
  auto newMod = b.create<hw::HWModuleOp>(
      op.getLoc(),
      b.getStringAttr(getVerilogModuleNameAttr(op).getValue() + suffix), ports);
  if (path)
    newMod->setAttr("output_file", path);
  newMod.setCommentAttr(b.getStringAttr("VCS coverage exclude_file"));

  // Update the mapping from old values to cloned values
  for (auto &port : llvm::enumerate(realInputs)) {
    cutMap.map(port.value(), newMod.getBody().getArgument(port.index()));
    for (auto extra : realReads[port.value()])
      cutMap.map(extra, newMod.getBody().getArgument(port.index()));
  }
  cutMap.map(op.getBodyBlock(), newMod.getBodyBlock());

  // Add an instance in the old module for the extracted module
  b = OpBuilder::atBlockTerminator(op.getBodyBlock());
  auto inst = b.create<hw::InstanceOp>(
      op.getLoc(), newMod, newMod.getName(), realInputs, ArrayAttr(),
      b.getStringAttr(
          ("__ETC_" + getVerilogModuleNameAttr(op).getValue() + suffix).str()));
  inst->setAttr("doNotPrint", b.getBoolAttr(true));
  b = OpBuilder::atBlockEnd(
      &op->getParentOfType<mlir::ModuleOp>()->getRegion(0).front());

  auto bindOp = b.create<sv::BindOp>(op.getLoc(), op.getNameAttr(),
                                     inst.getInnerSymAttr());
  if (fileName)
    bindOp->setAttr("output_file", fileName);
  return newMod;
}

// Some blocks have terminators, some don't
static void setInsertPointToEndOrTerminator(OpBuilder &builder, Block *block) {
  if (!block->empty() && isa<hw::HWModuleOp>(block->getParentOp()))
    builder.setInsertionPoint(&block->back());
  else
    builder.setInsertionPointToEnd(block);
}

// Shallow clone, which we use to not clone the content of blocks, doesn't
// clone the regions, so create all the blocks we need and update the mapping.
static void addBlockMapping(BlockAndValueMapping &cutMap, Operation *oldOp,
                            Operation *newOp) {
  assert(oldOp->getNumRegions() == newOp->getNumRegions());
  for (size_t i = 0, e = oldOp->getNumRegions(); i != e; ++i) {
    auto &oldRegion = oldOp->getRegion(i);
    auto &newRegion = newOp->getRegion(i);
    for (auto oi = oldRegion.begin(), oe = oldRegion.end(); oi != oe; ++oi) {
      cutMap.map(&*oi, &newRegion.emplaceBlock());
    }
  }
}

static bool hasOoOArgs(hw::HWModuleOp newMod, Operation *op) {
  for (auto arg : op->getOperands()) {
    auto argOp = arg.getDefiningOp(); // may be null
    if (!argOp)
      continue;
    if (argOp->getParentOfType<hw::HWModuleOp>() != newMod)
      return true;
  }
  return false;
}

// Do the cloning, which is just a pre-order traversal over the module looking
// for marked ops.
static void migrateOps(hw::HWModuleOp oldMod, hw::HWModuleOp newMod,
                       SetVector<Operation *> &depOps,
                       BlockAndValueMapping &cutMap) {
  SmallVector<Operation *, 16> lateBoundOps;
  OpBuilder b = OpBuilder::atBlockBegin(newMod.getBodyBlock());
  oldMod.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (depOps.count(op)) {
      setInsertPointToEndOrTerminator(b, cutMap.lookup(op->getBlock()));
      auto newOp = b.cloneWithoutRegions(*op, cutMap);
      addBlockMapping(cutMap, op, newOp);
      if (hasOoOArgs(newMod, newOp))
        lateBoundOps.push_back(newOp);
    }
  });
  // update any operand which was emitted before it's defining op was.
  for (auto op : lateBoundOps)
    for (unsigned argidx = 0, e = op->getNumOperands(); argidx < e; ++argidx) {
      Value arg = op->getOperand(argidx);
      if (cutMap.contains(arg))
        op->setOperand(argidx, cutMap.lookup(arg));
    }
}

// Check if the module has already been bound.
static bool isBound(hw::HWModuleLike op, hw::InstanceGraph &instanceGraph) {
  auto *node = instanceGraph.lookup(op);
  return llvm::any_of(node->uses(), [](hw::InstanceRecord *a) {
    auto inst = a->getInstance();
    if (!inst)
      return false;
    return inst->hasAttr("doNotPrint");
  });
}

// Move any old modules that are test code only to the test code area.
static void maybeMoveToTestCode(hw::HWModuleOp oldMod, Attribute testBenchDir,
                                Attribute bindFile,
                                hw::InstanceGraph &instanceGraph) {
  // Ensure we have a valid test code path.
  if (!testBenchDir)
    return;

  // Check if the module only has inputs.
  if (oldMod.getNumOutputs() != 0)
    return;

  // Check if this module has been bound already, and return early if so. This
  // can happen, for example, if a module is input only, but exists to feed both
  // asserts and covers. In such cases, the binds end up in the bind file for
  // the first kind of test code statement we saw, and we expect this to not
  // cause issues.
  if (isBound(oldMod, instanceGraph)) {
    LLVM_DEBUG(oldMod->emitWarning("already bound, skipping for ") << bindFile);
    return;
  }

  // Move the module to the test code path.
  oldMod->setAttr("output_file", testBenchDir);

  // Replace its instances with binds.
  hw::InstanceGraphNode *node = instanceGraph.lookup(oldMod);
  OpBuilder b = OpBuilder(oldMod.getContext());
  for (hw::InstanceRecord *use : node->uses()) {
    OpBuilder::InsertionGuard g(b);

    hw::HWInstanceLike instLike = use->getInstance();
    if (!instLike)
      continue;

    // Mark the instance to not be printed.
    hw::InstanceOp inst = cast<hw::InstanceOp>(instLike.getOperation());
    hw::HWModuleLike parent = use->getParent()->getModule();
    inst->setAttr("doNotPrint", b.getBoolAttr(true));

    // Give the instance an inner sym if it needs one.
    if (!inst.getInnerSym().has_value())
      inst.setInnerSymAttr(
          b.getStringAttr("__" + inst.getInstanceName() + "__"));

    // Create the bind.
    b.setInsertionPointToEnd(
        &oldMod->getParentOfType<mlir::ModuleOp>()->getRegion(0).front());
    auto bindOp = b.create<sv::BindOp>(inst.getLoc(), parent.moduleNameAttr(),
                                       inst.getInnerSymAttr());
    if (bindFile)
      bindOp->setAttr("output_file", bindFile);
  }
}

//===----------------------------------------------------------------------===//
// StubExternalModules Pass
//===----------------------------------------------------------------------===//

namespace {

struct SVExtractTestCodeImplPass
    : public SVExtractTestCodeBase<SVExtractTestCodeImplPass> {
  void runOnOperation() override;

private:
  void doModule(hw::HWModuleOp module, std::function<bool(Operation *)> fn,
                StringRef suffix, Attribute path, Attribute bindFile,
                Attribute testBenchDir, hw::InstanceGraph &instanceGraph) {
    bool hasError = false;
    // Find Operations of interest.
    SetVector<Operation *> roots;
    module->walk([&fn, &roots, &hasError](Operation *op) {
      if (fn(op)) {
        roots.insert(op);
        if (op->getNumResults()) {
          op->emitError("Extracting op with result");
          hasError = true;
        }
      }
    });
    if (hasError) {
      signalPassFailure();
      return;
    }
    // No Ops?  No problem.
    if (roots.empty())
      return;

    // Find the data-flow and structural ops to clone.  Result includes roots.
    auto opsToClone = computeCloneSet(roots);
    numOpsExtracted += opsToClone.size();

    // Find the dataflow into the clone set
    SetVector<Value> inputs;
    for (auto op : opsToClone)
      for (auto arg : op->getOperands()) {
        auto argOp = arg.getDefiningOp(); // may be null
        if (!opsToClone.count(argOp))
          inputs.insert(arg);
      }

    // Find instances that only feed the clone set, and add them if possible.
    SmallPtrSet<Operation *, 32> opsToErase;
    addInstancesToCloneSet(inputs, opsToClone, opsToErase, extractedInstances);
    numOpsErased += opsToErase.size();

    // Make a module to contain the clone set, with arguments being the cut
    BlockAndValueMapping cutMap;
    auto bmod =
        createModuleForCut(module, inputs, cutMap, suffix, path, bindFile);
    // do the clone
    migrateOps(module, bmod, opsToClone, cutMap);
    // erase old operations of interest
    for (auto op : roots)
      op->erase();

    // Erase any instances that were extracted, and their forward dataflow.
    for (auto *op : opsToErase) {
      if (roots.contains(op))
        continue;
      op->dropAllUses();
      op->erase();
    }

    // Move any old modules that are test code only to the test code area.
    maybeMoveToTestCode(module, testBenchDir, bindFile, instanceGraph);
  }

  // Move any modules that had all instances extracted to the testbench path.
  void maybeMoveExtractedModules(hw::InstanceGraph &instanceGraph,
                                 Attribute testBenchDir) {
    // Ensure we have a valid test code path.
    if (!testBenchDir)
      return;

    // Check each module that had instances extracted.
    for (auto &pair : extractedInstances) {
      hw::InstanceGraphNode *node = instanceGraph.lookup(pair.first);

      // See if all instances were extracted.
      bool allInstancesExtracted = true;
      for (hw::InstanceRecord *use : node->uses()) {
        allInstancesExtracted &= extractedInstances[pair.first].contains(
            use->getInstance().getOperation());
      }

      if (!allInstancesExtracted)
        continue;

      // If so, move the module to the test code path.
      hw::HWModuleLike mod = node->getModule();
      mod->setAttr("output_file", testBenchDir);
    }
  }

  // Map from module name to set of extracted instances for that module.
  DenseMap<StringAttr, SmallPtrSet<Operation *, 32>> extractedInstances;
};

} // end anonymous namespace

void SVExtractTestCodeImplPass::runOnOperation() {
  auto top = getOperation();
  auto *topLevelModule = top.getBody();
  auto assertDir =
      top->getAttrOfType<hw::OutputFileAttr>("firrtl.extract.assert");
  auto assumeDir =
      top->getAttrOfType<hw::OutputFileAttr>("firrtl.extract.assume");
  auto coverDir =
      top->getAttrOfType<hw::OutputFileAttr>("firrtl.extract.cover");
  auto testBenchDir =
      top->getAttrOfType<hw::OutputFileAttr>("firrtl.extract.testbench");
  auto assertBindFile =
      top->getAttrOfType<hw::OutputFileAttr>("firrtl.extract.assert.bindfile");
  auto assumeBindFile =
      top->getAttrOfType<hw::OutputFileAttr>("firrtl.extract.assume.bindfile");
  auto coverBindFile =
      top->getAttrOfType<hw::OutputFileAttr>("firrtl.extract.cover.bindfile");

  hw::HWSymbolCache symCache;
  symCache.addDefinitions(top);
  symCache.freeze();

  // Symbols not in the cache will only be fore instances added by an extract
  // phase and are not instances that could possibly have extract flags on them.
  auto isAssert = [&symCache](Operation *op) -> bool {
    if (auto inst = dyn_cast<hw::InstanceOp>(op))
      if (auto mod = symCache.getDefinition(inst.getModuleNameAttr()))
        if (mod->getAttr("firrtl.extract.assert.extra"))
          return true;

    // If the format of assert is "ifElseFatal", PrintOp is lowered into
    // ErrorOp. So we have to check message contents whether they encode
    // verifications. See FIRParserAsserts for more details.
    if (auto error = dyn_cast<ErrorOp>(op)) {
      if (auto message = error.getMessage())
        return message->startswith("assert:") ||
               message->startswith("Assertion failed") ||
               message->startswith("assertNotX:") ||
               message->contains("[verif-library-assert]");
      return false;
    }

    return isa<AssertOp>(op) || isa<FinishOp>(op) || isa<FWriteOp>(op) ||
           isa<AssertConcurrentOp>(op) || isa<FatalOp>(op);
  };
  auto isAssume = [&symCache](Operation *op) -> bool {
    if (auto inst = dyn_cast<hw::InstanceOp>(op))
      if (auto mod = symCache.getDefinition(inst.getModuleNameAttr()))
        if (mod->getAttr("firrtl.extract.assume.extra"))
          return true;
    return isa<AssumeOp>(op) || isa<AssumeConcurrentOp>(op);
  };
  auto isCover = [&symCache](Operation *op) -> bool {
    if (auto inst = dyn_cast<hw::InstanceOp>(op))
      if (auto mod = symCache.getDefinition(inst.getModuleNameAttr()))
        if (mod->getAttr("firrtl.extract.cover.extra"))
          return true;
    return isa<CoverOp>(op) || isa<CoverConcurrentOp>(op);
  };

  auto &instanceGraph = getAnalysis<circt::hw::InstanceGraph>();

  for (auto &op : topLevelModule->getOperations()) {
    if (auto rtlmod = dyn_cast<hw::HWModuleOp>(op)) {
      // Extract two sets of ops to different modules.  This will add modules,
      // but not affect modules in the symbol table.  If any instance of the
      // module is bound, then extraction is skipped.  This avoids problems
      // where certain simulators dislike having binds that target bound
      // modules.
      if (isBound(rtlmod, instanceGraph))
        continue;

      // In the module is in test harness, we don't have to extract from it.
      if (rtlmod->hasAttr("firrtl.extract.do_not_extract")) {
        rtlmod->removeAttr("firrtl.extract.do_not_extract");
        continue;
      }

      doModule(rtlmod, isAssert, "_assert", assertDir, assertBindFile,
               testBenchDir, instanceGraph);
      doModule(rtlmod, isAssume, "_assume", assumeDir, assumeBindFile,
               testBenchDir, instanceGraph);
      doModule(rtlmod, isCover, "_cover", coverDir, coverBindFile, testBenchDir,
               instanceGraph);
    }
  }

  // After all instances are processed, move any modules that had all instances
  // extracted to the testbench path.
  maybeMoveExtractedModules(instanceGraph, testBenchDir);

  // We have to wait until all the instances are processed to clean up the
  // annotations.
  for (auto &op : topLevelModule->getOperations())
    if (isa<hw::HWModuleOp, hw::HWModuleExternOp>(op)) {
      op.removeAttr("firrtl.extract.assert.extra");
      op.removeAttr("firrtl.extract.cover.extra");
      op.removeAttr("firrtl.extract.assume.extra");
    }
  top->removeAttr("firrtl.extract.testbench");
}

std::unique_ptr<Pass> circt::sv::createSVExtractTestCodePass() {
  return std::make_unique<SVExtractTestCodeImplPass>();
}
