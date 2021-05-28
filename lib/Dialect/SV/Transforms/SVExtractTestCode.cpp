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

#include "SVPassDetail.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Dialect/SV/SVVisitors.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"

#include <set>

using namespace mlir;
using namespace circt;
using namespace sv;

//===----------------------------------------------------------------------===//
// StubExternalModules Helpers
//===----------------------------------------------------------------------===//

// Compute the dataflow for a set of ops.
static void dataflowSlice(ArrayRef<Operation *> ops,
                          SetVector<Operation *> &results) {
  for (auto op : ops) {
    getBackwardSlice(op, &results, [](Operation *testOp) -> bool {
      return !isa<sv::ReadInOutOp>(testOp) && !isa<hw::InstanceOp>(testOp) &&
             !isa<sv::PAssignOp>(testOp) && !isa<sv::BPAssignOp>(testOp);
    });
  }
}

// Compute the ops defining the blocks a set of ops are in.
static void blockSlice(ArrayRef<Operation *> ops,
                       SetVector<Operation *> &blocks) {
  for (auto op : ops) {
    while (!isa<hw::HWModuleOp>(op->getParentOp())) {
      op = op->getParentOp();
      blocks.insert(op);
    }
  }
}

// Aggressively mark operations to be moved to the new module.  This leaves
// maximum flexibility for optimization after removal of the nodes from the old
// module.
static SetVector<Operation *> computeCloneSet(ArrayRef<Operation *> roots) {
  SetVector<Operation *> results;
  // Get Dataflow for roots
  dataflowSlice(roots, results);

  // Get Blocks
  SetVector<Operation *> blocks;
  blockSlice(roots, blocks);
  blockSlice(results.getArrayRef(), blocks);

  // Make sure dataflow to block args (if conds, etc) is included
  dataflowSlice(blocks.getArrayRef(), results);

  // include the blocks and roots to clone
  results.insert(roots.begin(), roots.end());
  results.insert(blocks.begin(), blocks.end());

  return results;
}

// Given a set of values, construct a module and bind instance of that module
// that passes those values through.  Returns the new module and the instance
// pointing to it.
static hw::HWModuleOp createModuleForCut(hw::HWModuleOp op,
                                         SetVector<Value> &inputs,
                                         BlockAndValueMapping &cutMap,
                                         StringRef suffix, StringRef path) {
  OpBuilder b(op->getParentOfType<mlir::ModuleOp>()->getRegion(0));

  // Construct the ports, this is just the input Values
  SmallVector<hw::ModulePortInfo> ports;
  for (auto port : llvm::enumerate(inputs))
    ports.push_back(
        {b.getStringAttr(""), hw::INPUT, port.value().getType(), port.index()});

  // Create the module, setting the output path if indicated
  std::array<NamedAttribute, 1> pathAttr = {
      b.getNamedAttr("outputPath", b.getStringAttr(path))};
  auto newMod = b.create<hw::HWModuleOp>(
      op.getLoc(),
      b.getStringAttr((getVerilogModuleNameAttr(op).getValue() + suffix).str()),
      ports, path.empty() ? ArrayRef<NamedAttribute>() : pathAttr);

  // Update the mapping from old values to cloned values
  for (auto port : llvm::enumerate(inputs))
    cutMap.map(port.value(), newMod.body().getArgument(port.index()));
  cutMap.map(op.getBodyBlock(), newMod.getBodyBlock());

  // Add an instance in the old module for the extracted module
  b = OpBuilder::atBlockTerminator(op.getBodyBlock());
  auto inst = b.create<hw::InstanceOp>(
      op.getLoc(), ArrayRef<Type>(), ("InvisibleBind" + suffix).str(),
      newMod.getName(), inputs.getArrayRef(), DictionaryAttr(),
      b.getStringAttr(
          ("__ETC_" + getVerilogModuleNameAttr(op).getValue() + suffix).str()));
  inst->setAttr("doNotPrint", b.getBoolAttr(true));
  b = OpBuilder::atBlockEnd(
      &op->getParentOfType<mlir::ModuleOp>()->getRegion(0).front());
  b.create<sv::BindOp>(op.getLoc(), b.getSymbolRefAttr(inst));
  return newMod;
}

// Some blocks have terminators, some don't
static void setInsertPointToEndOrTerminator(OpBuilder &builder, Block *block) {
  if (!block->empty() && block->back().mightHaveTrait<OpTrait::IsTerminator>())
    builder.setInsertionPoint(&block->back());
  else
    builder.setInsertionPointToEnd(block);
}

// Shallow clone, which we use to not clone the content of blocks, doesn't clone
// the regions, so create all the blocks we need and update the mapping.
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

// Do the cloning, which is just a pre-order traversal over the module looking
// for marked ops.
static void migrateOps(hw::HWModuleOp oldMod, hw::HWModuleOp newMod,
                       SetVector<Operation *> &depOps,
                       BlockAndValueMapping &cutMap) {
  OpBuilder b = OpBuilder::atBlockBegin(newMod.getBodyBlock());
  oldMod.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (depOps.count(op)) {
      setInsertPointToEndOrTerminator(b, cutMap.lookup(op->getBlock()));
      auto newOp = b.cloneWithoutRegions(*op, cutMap);
      addBlockMapping(cutMap, op, newOp);
    }
  });
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
                StringRef suffix, StringRef path) {
    // Find Operations of interest.
    SmallVector<Operation *> roots;
    module->walk([&fn, &roots](Operation *op) {
      if (fn(op))
        roots.push_back(op);
    });
    // No Ops?  No problem.
    if (roots.empty())
      return;
    // Find the data-flow and structural ops to clone.  Result includes roots.
    auto opsToClone = computeCloneSet(roots);
    // Find the dataflow into the clone set
    SetVector<Value> inputs;
    for (auto op : opsToClone)
      for (auto arg : op->getOperands()) {
        auto argOp = arg.getDefiningOp(); // may be null
        if (!opsToClone.count(argOp))
          inputs.insert(arg);
      }

    // Make a module to contain the clone set, with arguments being the cut
    BlockAndValueMapping cutMap;
    auto bmod = createModuleForCut(module, inputs, cutMap, suffix, path);
    // do the clone
    migrateOps(module, bmod, opsToClone, cutMap);
    // erase old operations of interest
    for (auto op : roots)
      op->erase();
  }
};

} // end anonymous namespace

void SVExtractTestCodeImplPass::runOnOperation() {
  auto *topLevelModule = getOperation().getBody();
  for (auto &op : topLevelModule->getOperations())
    if (auto rtlmod = dyn_cast<hw::HWModuleOp>(op)) {
      // Extract two sets of ops to different modules
      doModule(
          rtlmod,
          [](Operation *op) -> bool {
            return isa<AssertOp>(op) || isa<AssumeOp>(op) ||
                   isa<FinishOp>(op) || isa<FWriteOp>(op);
          },
          "_assert", "generated/asserts");
      doModule(
          rtlmod, [](Operation *op) -> bool { return isa<CoverOp>(op); },
          "_cover", "generated/covers/*  */");
    }
}

std::unique_ptr<Pass> circt::sv::createSVExtractTestCodePass() {
  return std::make_unique<SVExtractTestCodeImplPass>();
}
