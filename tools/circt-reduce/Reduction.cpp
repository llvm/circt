//===- Reduction.cpp - Reductions for circt-reduce ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines abstract reduction patterns for the 'circt-reduce' tool.
//
//===----------------------------------------------------------------------===//

#include "Reduction.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/InitAllDialects.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Reducer/Tester.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "circt-reduce"

using namespace llvm;
using namespace mlir;
using namespace circt;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

/// A utility doing lazy construction of `SymbolTable`s and `SymbolUserMap`s,
/// which is handy for reductions that need to look up a lot of symbols.
struct SymbolCache {
  SymbolTable &getSymbolTable(Operation *op) {
    return tables.getSymbolTable(op);
  }
  SymbolTable &getNearestSymbolTable(Operation *op) {
    return getSymbolTable(SymbolTable::getNearestSymbolTable(op));
  }

  SymbolUserMap &getSymbolUserMap(Operation *op) {
    auto it = userMaps.find(op);
    if (it != userMaps.end())
      return it->second;
    return userMaps.insert({op, SymbolUserMap(tables, op)}).first->second;
  }
  SymbolUserMap &getNearestSymbolUserMap(Operation *op) {
    return getSymbolUserMap(SymbolTable::getNearestSymbolTable(op));
  }

  void clear() {
    tables = SymbolTableCollection();
    userMaps.clear();
  }

private:
  SymbolTableCollection tables;
  SmallDenseMap<Operation *, SymbolUserMap, 2> userMaps;
};

/// Utility to easily get the instantiated firrtl::FModuleOp or an empty
/// optional in case another type of module is instantiated.
static std::optional<firrtl::FModuleOp>
findInstantiatedModule(firrtl::InstanceOp instOp, SymbolCache &symbols) {
  auto *tableOp = SymbolTable::getNearestSymbolTable(instOp);
  auto moduleOp = dyn_cast<firrtl::FModuleOp>(
      instOp.getReferencedModule(symbols.getSymbolTable(tableOp))
          .getOperation());
  return moduleOp ? std::optional(moduleOp) : std::nullopt;
}

/// Compute the number of operations in a module. Recursively add the number of
/// operations in instantiated modules.
/// @param countMultipleInstantiations: If a module is instantiated multiple
/// times and this flag is false, count it only once (to better represent code
/// size reduction rather than area reduction of the actual hardware).
/// @param countElsewhereInstantiated: If a module is also instantiated in
/// another subtree of the design then don't count it if this flag is false.
static uint64_t computeTransitiveModuleSize(
    SmallVector<std::pair<firrtl::FModuleOp, uint64_t>> &modules,
    SmallVector<Operation *> &instances, bool countMultipleInstantiations,
    bool countElsewhereInstantiated) {
  std::sort(instances.begin(), instances.end());
  std::sort(modules.begin(), modules.end(),
            [](auto a, auto b) { return a.first < b.first; });

  auto *end = modules.end();
  if (!countMultipleInstantiations)
    end = std::unique(modules.begin(), modules.end(),
                      [](auto a, auto b) { return a.first == b.first; });

  uint64_t totalOperations = 0;

  for (auto *iter = modules.begin(); iter != end; ++iter) {
    auto moduleOp = iter->first;

    auto allInstancesCovered = [&]() {
      return llvm::all_of(
          *moduleOp.getSymbolUses(moduleOp->getParentOfType<ModuleOp>()),
          [&](auto symbolUse) {
            return std::binary_search(instances.begin(), instances.end(),
                                      symbolUse.getUser());
          });
    };

    if (countElsewhereInstantiated || allInstancesCovered())
      totalOperations += iter->second;
  }

  return totalOperations;
}

static LogicalResult collectInstantiatedModules(
    std::optional<firrtl::FModuleOp> fmoduleOp, SymbolCache &symbols,
    SmallVector<std::pair<firrtl::FModuleOp, uint64_t>> &modules,
    SmallVector<Operation *> &instances) {
  if (!fmoduleOp)
    return failure();

  uint64_t opCount = 0;
  WalkResult result = fmoduleOp->walk([&](Operation *op) {
    if (auto instOp = dyn_cast<firrtl::InstanceOp>(op)) {
      auto moduleOp = findInstantiatedModule(instOp, symbols);
      if (!moduleOp) {
        LLVM_DEBUG(llvm::dbgs()
                   << "- `" << fmoduleOp->moduleName()
                   << "` recursively instantiated non-FIRRTL module.\n");
        return WalkResult::interrupt();
      }

      if (failed(collectInstantiatedModules(moduleOp, symbols, modules,
                                            instances)))
        return WalkResult::interrupt();

      instances.push_back(instOp);
    }

    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();

  modules.push_back(std::make_pair(*fmoduleOp, opCount));

  return success();
}

/// Check that all connections to a value are invalids.
static bool onlyInvalidated(Value arg) {
  return llvm::all_of(arg.getUses(), [](OpOperand &use) {
    auto *op = use.getOwner();
    if (!isa<firrtl::ConnectOp, firrtl::StrictConnectOp>(op))
      return false;
    if (use.getOperandNumber() != 0)
      return false;
    if (!op->getOperand(1).getDefiningOp<firrtl::InvalidValueOp>())
      return false;
    return true;
  });
}

/// A tracker for track NLAs affected by a reduction. Performs the necessary
/// cleanup steps in order to maintain IR validity after the reduction has
/// applied. For example, removing an instance that forms part of an NLA path
/// requires that NLA to be removed as well.
struct NLARemover {
  /// Clear the set of marked NLAs. Call this before attempting a reduction.
  void clear() { nlasToRemove.clear(); }

  /// Remove all marked annotations. Call this after applying a reduction in
  /// order to validate the IR.
  void remove(mlir::ModuleOp module) {
    unsigned numRemoved = 0;
    for (Operation &rootOp : *module.getBody()) {
      if (!isa<firrtl::CircuitOp>(&rootOp))
        continue;
      SymbolTable symbolTable(&rootOp);
      for (auto sym : nlasToRemove) {
        if (auto *op = symbolTable.lookup(sym)) {
          ++numRemoved;
          op->erase();
        }
      }
    }
    LLVM_DEBUG({
      unsigned numLost = nlasToRemove.size() - numRemoved;
      if (numRemoved > 0 || numLost > 0) {
        llvm::dbgs() << "Removed " << numRemoved << " NLAs";
        if (numLost > 0)
          llvm::dbgs() << " (" << numLost << " no longer there)";
        llvm::dbgs() << "\n";
      }
    });
  }

  /// Mark all NLAs referenced in the given annotation as to be removed. This
  /// can be an entire array or dictionary of annotations, and the function will
  /// descend into child annotations appropriately.
  void markNLAsInAnnotation(Attribute anno) {
    if (auto dict = anno.dyn_cast<DictionaryAttr>()) {
      if (auto field = dict.getAs<FlatSymbolRefAttr>("circt.nonlocal"))
        nlasToRemove.insert(field.getAttr());
      for (auto namedAttr : dict)
        markNLAsInAnnotation(namedAttr.getValue());
    } else if (auto array = anno.dyn_cast<ArrayAttr>()) {
      for (auto attr : array)
        markNLAsInAnnotation(attr);
    }
  }

  /// Mark all NLAs referenced in an operation. Also traverses all nested
  /// operations. Call this before removing an operation, to mark any associated
  /// NLAs as to be removed as well.
  void markNLAsInOperation(Operation *op) {
    op->walk([&](Operation *op) {
      if (auto annos = op->getAttrOfType<ArrayAttr>("annotations"))
        markNLAsInAnnotation(annos);
    });
  }

  /// The set of NLAs to remove, identified by their symbol.
  llvm::DenseSet<StringAttr> nlasToRemove;
};

//===----------------------------------------------------------------------===//
// Reduction
//===----------------------------------------------------------------------===//

Reduction::~Reduction() {}

//===----------------------------------------------------------------------===//
// Pass Reduction
//===----------------------------------------------------------------------===//

PassReduction::PassReduction(MLIRContext *context, std::unique_ptr<Pass> pass,
                             bool canIncreaseSize, bool oneShot)
    : context(context), canIncreaseSize(canIncreaseSize), oneShot(oneShot) {
  passName = pass->getArgument();
  if (passName.empty())
    passName = pass->getName();

  pm = std::make_unique<PassManager>(context, "builtin.module",
                                     mlir::OpPassManager::Nesting::Explicit);
  auto opName = pass->getOpName();
  if (opName && opName->equals("firrtl.circuit"))
    pm->nest<firrtl::CircuitOp>().addPass(std::move(pass));
  else if (opName && opName->equals("firrtl.module"))
    pm->nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
        std::move(pass));
  else
    pm->nest<mlir::ModuleOp>().addPass(std::move(pass));
}

uint64_t PassReduction::match(Operation *op) {
  return op->getName() == pm->getOpName(*context);
}

LogicalResult PassReduction::rewrite(Operation *op) { return pm->run(op); }

std::string PassReduction::getName() const { return passName.str(); }

//===----------------------------------------------------------------------===//
// Concrete Sample Reductions (to later move into the dialects)
//===----------------------------------------------------------------------===//

/// A sample reduction pattern that maps `firrtl.module` to `firrtl.extmodule`.
struct ModuleExternalizer : public Reduction {
  void beforeReduction(mlir::ModuleOp op) override {
    nlaRemover.clear();
    symbols.clear();
  }
  void afterReduction(mlir::ModuleOp op) override { nlaRemover.remove(op); }

  uint64_t match(Operation *op) override {
    if (auto fmoduleOp = dyn_cast<firrtl::FModuleOp>(op)) {
      SmallVector<std::pair<firrtl::FModuleOp, uint64_t>> modules;
      SmallVector<Operation *> instances;
      if (failed(collectInstantiatedModules(fmoduleOp, symbols, modules,
                                            instances)))
        return 0;
      return computeTransitiveModuleSize(modules, instances,
                                         /*countMultipleInstantiations=*/false,
                                         /*countElsewhereInstantiated=*/true);
    }
    return 0;
  }

  LogicalResult rewrite(Operation *op) override {
    auto module = cast<firrtl::FModuleOp>(op);
    nlaRemover.markNLAsInOperation(op);
    OpBuilder builder(module);
    builder.create<firrtl::FExtModuleOp>(
        module->getLoc(),
        module->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()),
        module.getPorts(), StringRef(), module.getAnnotationsAttr());
    module->erase();
    return success();
  }

  std::string getName() const override { return "module-externalizer"; }

  SymbolCache symbols;
  NLARemover nlaRemover;
};

/// Invalidate all the leaf fields of a value with a given flippedness by
/// connecting an invalid value to them. This is useful for ensuring that all
/// output ports of an instance or memory (including those nested in bundles)
/// are properly invalidated.
static void invalidateOutputs(ImplicitLocOpBuilder &builder, Value value,
                              SmallDenseMap<Type, Value, 8> &invalidCache,
                              bool flip = false) {
  auto type = value.getType().dyn_cast<firrtl::FIRRTLType>();
  if (!type)
    return;

  // Descend into bundles by creating subfield ops.
  if (auto bundleType = type.dyn_cast<firrtl::BundleType>()) {
    for (auto &element : llvm::enumerate(bundleType.getElements())) {
      auto subfield =
          builder.createOrFold<firrtl::SubfieldOp>(value, element.index());
      invalidateOutputs(builder, subfield, invalidCache,
                        flip ^ element.value().isFlip);
      if (subfield.use_empty())
        subfield.getDefiningOp()->erase();
    }
    return;
  }

  // Descend into vectors by creating subindex ops.
  if (auto vectorType = type.dyn_cast<firrtl::FVectorType>()) {
    for (unsigned i = 0, e = vectorType.getNumElements(); i != e; ++i) {
      auto subindex = builder.createOrFold<firrtl::SubindexOp>(value, i);
      invalidateOutputs(builder, subindex, invalidCache, flip);
      if (subindex.use_empty())
        subindex.getDefiningOp()->erase();
    }
    return;
  }

  // Only drive outputs.
  if (flip)
    return;
  Value invalid = invalidCache.lookup(type);
  if (!invalid) {
    invalid = builder.create<firrtl::InvalidValueOp>(type);
    invalidCache.insert({type, invalid});
  }
  builder.create<firrtl::ConnectOp>(value, invalid);
}

/// Connect a value to every leave of a destination value.
static void connectToLeafs(ImplicitLocOpBuilder &builder, Value dest,
                           Value value) {
  auto type = dest.getType().dyn_cast<firrtl::FIRRTLBaseType>();
  if (!type)
    return;
  if (auto bundleType = type.dyn_cast<firrtl::BundleType>()) {
    for (auto &element : llvm::enumerate(bundleType.getElements()))
      connectToLeafs(builder,
                     builder.create<firrtl::SubfieldOp>(dest, element.index()),
                     value);
    return;
  }
  if (auto vectorType = type.dyn_cast<firrtl::FVectorType>()) {
    for (unsigned i = 0, e = vectorType.getNumElements(); i != e; ++i)
      connectToLeafs(builder, builder.create<firrtl::SubindexOp>(dest, i),
                     value);
    return;
  }
  auto valueType = value.getType().dyn_cast<firrtl::FIRRTLBaseType>();
  if (!valueType)
    return;
  auto destWidth = type.getBitWidthOrSentinel();
  auto valueWidth = valueType ? valueType.getBitWidthOrSentinel() : -1;
  if (destWidth >= 0 && valueWidth >= 0 && destWidth < valueWidth)
    value = builder.create<firrtl::HeadPrimOp>(value, destWidth);
  if (!type.isa<firrtl::UIntType>()) {
    if (type.isa<firrtl::SIntType>())
      value = builder.create<firrtl::AsSIntPrimOp>(value);
    else
      return;
  }
  builder.create<firrtl::ConnectOp>(dest, value);
}

/// Reduce all leaf fields of a value through an XOR tree.
static void reduceXor(ImplicitLocOpBuilder &builder, Value &into, Value value) {
  auto type = value.getType().dyn_cast<firrtl::FIRRTLType>();
  if (!type)
    return;
  if (auto bundleType = type.dyn_cast<firrtl::BundleType>()) {
    for (auto &element : llvm::enumerate(bundleType.getElements()))
      reduceXor(
          builder, into,
          builder.createOrFold<firrtl::SubfieldOp>(value, element.index()));
    return;
  }
  if (auto vectorType = type.dyn_cast<firrtl::FVectorType>()) {
    for (unsigned i = 0, e = vectorType.getNumElements(); i != e; ++i)
      reduceXor(builder, into,
                builder.createOrFold<firrtl::SubindexOp>(value, i));
    return;
  }
  if (!type.isa<firrtl::UIntType>()) {
    if (type.isa<firrtl::SIntType>())
      value = builder.create<firrtl::AsUIntPrimOp>(value);
    else
      return;
  }
  into = into ? builder.createOrFold<firrtl::XorPrimOp>(into, value) : value;
}

/// A sample reduction pattern that maps `firrtl.instance` to a set of
/// invalidated wires. This often shortcuts a long iterative process of connect
/// invalidation, module externalization, and wire stripping
struct InstanceStubber : public Reduction {
  void beforeReduction(mlir::ModuleOp op) override {
    erasedInsts.clear();
    erasedModules.clear();
    symbols.clear();
    nlaRemover.clear();
  }
  void afterReduction(mlir::ModuleOp op) override {
    // Look into deleted modules to find additional instances that are no longer
    // instantiated anywhere.
    SmallVector<Operation *> worklist;
    auto deadInsts = erasedInsts;
    for (auto *op : erasedModules)
      worklist.push_back(op);
    while (!worklist.empty()) {
      auto *op = worklist.pop_back_val();
      auto *tableOp = SymbolTable::getNearestSymbolTable(op);
      op->walk([&](firrtl::InstanceOp instOp) {
        auto moduleOp =
            instOp.getReferencedModule(symbols.getSymbolTable(tableOp));
        deadInsts.insert(instOp);
        if (llvm::all_of(
                symbols.getSymbolUserMap(tableOp).getUsers(moduleOp),
                [&](Operation *user) { return deadInsts.contains(user); })) {
          LLVM_DEBUG(llvm::dbgs() << "- Removing transitively unused module `"
                                  << moduleOp.moduleName() << "`\n");
          erasedModules.insert(moduleOp);
          worklist.push_back(moduleOp);
        }
      });
    }

    for (auto *op : erasedInsts)
      op->erase();
    for (auto *op : erasedModules)
      op->erase();
    nlaRemover.remove(op);
  }

  uint64_t match(Operation *op) override {
    if (auto instOp = dyn_cast<firrtl::InstanceOp>(op)) {
      auto fmoduleOp = findInstantiatedModule(instOp, symbols);
      SmallVector<std::pair<firrtl::FModuleOp, uint64_t>> modules;
      SmallVector<Operation *> instances;
      if (failed(collectInstantiatedModules(fmoduleOp, symbols, modules,
                                            instances)))
        return 0;
      return computeTransitiveModuleSize(modules, instances,
                                         /*countMultipleInstantiations=*/false,
                                         /*countElsewhereInstantiated=*/false);
    }
    return 0;
  }

  LogicalResult rewrite(Operation *op) override {
    auto instOp = cast<firrtl::InstanceOp>(op);
    LLVM_DEBUG(llvm::dbgs()
               << "Stubbing instance `" << instOp.getName() << "`\n");
    ImplicitLocOpBuilder builder(instOp.getLoc(), instOp);
    SmallDenseMap<Type, Value, 8> invalidCache;
    for (unsigned i = 0, e = instOp.getNumResults(); i != e; ++i) {
      auto result = instOp.getResult(i);
      auto name = builder.getStringAttr(Twine(instOp.getName()) + "_" +
                                        instOp.getPortNameStr(i));
      auto wire = builder.create<firrtl::WireOp>(
          result.getType(), name, firrtl::NameKindEnum::DroppableName,
          instOp.getPortAnnotation(i), StringAttr{});
      invalidateOutputs(builder, wire, invalidCache,
                        instOp.getPortDirection(i) == firrtl::Direction::In);
      result.replaceAllUsesWith(wire);
    }
    auto tableOp = SymbolTable::getNearestSymbolTable(instOp);
    auto moduleOp = instOp.getReferencedModule(symbols.getSymbolTable(tableOp));
    nlaRemover.markNLAsInOperation(instOp);
    erasedInsts.insert(instOp);
    if (llvm::all_of(
            symbols.getSymbolUserMap(tableOp).getUsers(moduleOp),
            [&](Operation *user) { return erasedInsts.contains(user); })) {
      LLVM_DEBUG(llvm::dbgs() << "- Removing now unused module `"
                              << moduleOp.moduleName() << "`\n");
      erasedModules.insert(moduleOp);
    }
    return success();
  }

  std::string getName() const override { return "instance-stubber"; }
  bool acceptSizeIncrease() const override { return true; }

  SymbolCache symbols;
  NLARemover nlaRemover;
  llvm::DenseSet<Operation *> erasedInsts;
  llvm::DenseSet<Operation *> erasedModules;
};

/// A sample reduction pattern that maps `firrtl.mem` to a set of invalidated
/// wires.
struct MemoryStubber : public Reduction {
  void beforeReduction(mlir::ModuleOp op) override { nlaRemover.clear(); }
  void afterReduction(mlir::ModuleOp op) override { nlaRemover.remove(op); }
  uint64_t match(Operation *op) override { return isa<firrtl::MemOp>(op); }
  LogicalResult rewrite(Operation *op) override {
    auto memOp = cast<firrtl::MemOp>(op);
    LLVM_DEBUG(llvm::dbgs() << "Stubbing memory `" << memOp.getName() << "`\n");
    ImplicitLocOpBuilder builder(memOp.getLoc(), memOp);
    SmallDenseMap<Type, Value, 8> invalidCache;
    Value xorInputs;
    SmallVector<Value> outputs;
    for (unsigned i = 0, e = memOp.getNumResults(); i != e; ++i) {
      auto result = memOp.getResult(i);
      auto name = builder.getStringAttr(Twine(memOp.getName()) + "_" +
                                        memOp.getPortNameStr(i));
      auto wire = builder.create<firrtl::WireOp>(
          result.getType(), name, firrtl::NameKindEnum::DroppableName,
          memOp.getPortAnnotation(i), StringAttr{});
      invalidateOutputs(builder, wire, invalidCache, true);
      result.replaceAllUsesWith(wire);

      // Isolate the input and output data fields of the port.
      Value input, output;
      switch (memOp.getPortKind(i)) {
      case firrtl::MemOp::PortKind::Read:
        output = builder.createOrFold<firrtl::SubfieldOp>(wire, 3);
        break;
      case firrtl::MemOp::PortKind::Write:
        input = builder.createOrFold<firrtl::SubfieldOp>(wire, 3);
        break;
      case firrtl::MemOp::PortKind::ReadWrite:
        input = builder.createOrFold<firrtl::SubfieldOp>(wire, 5);
        output = builder.createOrFold<firrtl::SubfieldOp>(wire, 3);
        break;
      case firrtl::MemOp::PortKind::Debug:
        output = wire;
        break;
      }

      if (!result.getType().cast<firrtl::FIRRTLType>().isa<firrtl::RefType>()) {
        // Reduce all input ports to a single one through an XOR tree.
        unsigned numFields =
            wire.getType().cast<firrtl::BundleType>().getNumElements();
        for (unsigned i = 0; i != numFields; ++i) {
          if (i != 2 && i != 3 && i != 5)
            reduceXor(builder, xorInputs,
                      builder.createOrFold<firrtl::SubfieldOp>(wire, i));
        }
        if (input)
          reduceXor(builder, xorInputs, input);
      }

      // Track the output port to hook it up to the XORd input later.
      if (output)
        outputs.push_back(output);
    }

    // Hook up the outputs.
    for (auto output : outputs)
      connectToLeafs(builder, output, xorInputs);

    nlaRemover.markNLAsInOperation(memOp);
    memOp->erase();
    return success();
  }
  std::string getName() const override { return "memory-stubber"; }
  bool acceptSizeIncrease() const override { return true; }
  NLARemover nlaRemover;
};

/// Starting at the given `op`, traverse through it and its operands and erase
/// operations that have no more uses.
static void pruneUnusedOps(Operation *initialOp) {
  SmallVector<Operation *> worklist;
  SmallSet<Operation *, 4> handled;
  worklist.push_back(initialOp);
  while (!worklist.empty()) {
    auto op = worklist.pop_back_val();
    if (!op->use_empty())
      continue;
    for (auto arg : op->getOperands())
      if (auto argOp = arg.getDefiningOp())
        if (handled.insert(argOp).second)
          worklist.push_back(argOp);
    op->erase();
  }
}

/// Check whether an operation interacts with flows in any way, which can make
/// replacement and operand forwarding harder in some cases.
static bool isFlowSensitiveOp(Operation *op) {
  return isa<firrtl::WireOp, firrtl::RegOp, firrtl::RegResetOp,
             firrtl::InstanceOp, firrtl::SubfieldOp, firrtl::SubindexOp,
             firrtl::SubaccessOp>(op);
}

/// A sample reduction pattern that replaces all uses of an operation with one
/// of its operands. This can help pruning large parts of the expression tree
/// rapidly.
template <unsigned OpNum>
struct OperandForwarder : public Reduction {
  uint64_t match(Operation *op) override {
    if (op->getNumResults() != 1 || op->getNumOperands() < 2 ||
        OpNum >= op->getNumOperands())
      return 0;
    if (isFlowSensitiveOp(op))
      return 0;
    auto resultTy =
        op->getResult(0).getType().dyn_cast<firrtl::FIRRTLBaseType>();
    auto opTy =
        op->getOperand(OpNum).getType().dyn_cast<firrtl::FIRRTLBaseType>();
    return resultTy && opTy &&
           resultTy.getWidthlessType() == opTy.getWidthlessType() &&
           (resultTy.getBitWidthOrSentinel() == -1) ==
               (opTy.getBitWidthOrSentinel() == -1) &&
           resultTy.isa<firrtl::UIntType, firrtl::SIntType>();
  }
  LogicalResult rewrite(Operation *op) override {
    assert(match(op));
    ImplicitLocOpBuilder builder(op->getLoc(), op);
    auto result = op->getResult(0);
    auto operand = op->getOperand(OpNum);
    auto resultTy = result.getType().cast<firrtl::FIRRTLBaseType>();
    auto operandTy = operand.getType().cast<firrtl::FIRRTLBaseType>();
    auto resultWidth = resultTy.getBitWidthOrSentinel();
    auto operandWidth = operandTy.getBitWidthOrSentinel();
    Value newOp;
    if (resultWidth < operandWidth)
      newOp =
          builder.createOrFold<firrtl::BitsPrimOp>(operand, resultWidth - 1, 0);
    else if (resultWidth > operandWidth)
      newOp = builder.createOrFold<firrtl::PadPrimOp>(operand, resultWidth);
    else
      newOp = operand;
    LLVM_DEBUG(llvm::dbgs() << "Forwarding " << newOp << " in " << *op << "\n");
    result.replaceAllUsesWith(newOp);
    pruneUnusedOps(op);
    return success();
  }
  std::string getName() const override {
    return ("operand" + Twine(OpNum) + "-forwarder").str();
  }
};

/// A sample reduction pattern that replaces operations with a constant zero of
/// their type.
struct Constantifier : public Reduction {
  uint64_t match(Operation *op) override {
    if (op->getNumResults() != 1 || op->getNumOperands() == 0)
      return 0;
    if (isFlowSensitiveOp(op))
      return 0;
    auto type = op->getResult(0).getType().dyn_cast<firrtl::FIRRTLBaseType>();
    return type && type.isa<firrtl::UIntType, firrtl::SIntType>();
  }
  LogicalResult rewrite(Operation *op) override {
    assert(match(op));
    OpBuilder builder(op);
    auto type = op->getResult(0).getType().cast<firrtl::FIRRTLBaseType>();
    auto width = type.getBitWidthOrSentinel();
    if (width == -1)
      width = 64;
    auto newOp = builder.create<firrtl::ConstantOp>(
        op->getLoc(), type, APSInt(width, type.isa<firrtl::UIntType>()));
    op->replaceAllUsesWith(newOp);
    pruneUnusedOps(op);
    return success();
  }
  std::string getName() const override { return "constantifier"; }
};

/// A sample reduction pattern that replaces the right-hand-side of
/// `firrtl.connect` and `firrtl.strictconnect` operations with a
/// `firrtl.invalidvalue`. This removes uses from the fanin cone to these
/// connects and creates opportunities for reduction in DCE/CSE.
struct ConnectInvalidator : public Reduction {
  uint64_t match(Operation *op) override {
    if (!isa<firrtl::ConnectOp, firrtl::StrictConnectOp>(op))
      return false;
    auto type = op->getOperand(1).getType().dyn_cast<firrtl::FIRRTLBaseType>();
    return type && type.isPassive() &&
           !op->getOperand(1).getDefiningOp<firrtl::InvalidValueOp>();
  }
  LogicalResult rewrite(Operation *op) override {
    assert(match(op));
    auto rhs = op->getOperand(1);
    OpBuilder builder(op);
    auto invOp =
        builder.create<firrtl::InvalidValueOp>(rhs.getLoc(), rhs.getType());
    auto rhsOp = rhs.getDefiningOp();
    op->setOperand(1, invOp);
    if (rhsOp)
      pruneUnusedOps(rhsOp);
    return success();
  }
  std::string getName() const override { return "connect-invalidator"; }
  bool acceptSizeIncrease() const override { return true; }
};

/// A sample reduction pattern that removes operations which either produce no
/// results or their results have no users.
struct OperationPruner : public Reduction {
  void beforeReduction(mlir::ModuleOp op) override { symbols.clear(); }
  uint64_t match(Operation *op) override {
    return !isa<ModuleOp>(op) &&
           (op->getNumResults() == 0 || op->use_empty()) &&
           (!op->hasAttr(SymbolTable::getSymbolAttrName()) ||
            symbols.getNearestSymbolUserMap(op).useEmpty(op));
  }
  LogicalResult rewrite(Operation *op) override {
    assert(match(op));
    pruneUnusedOps(op);
    return success();
  }
  std::string getName() const override { return "operation-pruner"; }

  SymbolCache symbols;
};

/// A sample reduction pattern that removes FIRRTL annotations from ports and
/// operations.
struct AnnotationRemover : public Reduction {
  void beforeReduction(mlir::ModuleOp op) override { nlaRemover.clear(); }
  void afterReduction(mlir::ModuleOp op) override { nlaRemover.remove(op); }
  uint64_t match(Operation *op) override {
    return op->hasAttr("annotations") || op->hasAttr("portAnnotations");
  }
  LogicalResult rewrite(Operation *op) override {
    auto emptyArray = ArrayAttr::get(op->getContext(), {});
    if (auto annos = op->getAttr("annotations")) {
      nlaRemover.markNLAsInAnnotation(annos);
      op->setAttr("annotations", emptyArray);
    }
    if (auto annos = op->getAttr("portAnnotations")) {
      nlaRemover.markNLAsInAnnotation(annos);
      auto attr = emptyArray;
      if (isa<firrtl::InstanceOp>(op))
        attr = ArrayAttr::get(
            op->getContext(),
            SmallVector<Attribute>(op->getNumResults(), emptyArray));
      op->setAttr("portAnnotations", attr);
    }
    return success();
  }
  std::string getName() const override { return "annotation-remover"; }
  NLARemover nlaRemover;
};

/// A sample reduction pattern that removes ports from the root `firrtl.module`
/// if the port is not used or just invalidated.
struct RootPortPruner : public Reduction {
  uint64_t match(Operation *op) override {
    auto module = dyn_cast<firrtl::FModuleOp>(op);
    if (!module)
      return 0;
    auto circuit = module->getParentOfType<firrtl::CircuitOp>();
    if (!circuit)
      return 0;
    return circuit.getNameAttr() == module.getNameAttr();
  }
  LogicalResult rewrite(Operation *op) override {
    assert(match(op));
    auto module = cast<firrtl::FModuleOp>(op);
    size_t numPorts = module.getNumPorts();
    llvm::BitVector dropPorts(numPorts);
    for (unsigned i = 0; i != numPorts; ++i) {
      if (onlyInvalidated(module.getArgument(i))) {
        dropPorts.set(i);
        for (auto *user :
             llvm::make_early_inc_range(module.getArgument(i).getUsers()))
          user->erase();
      }
    }
    module.erasePorts(dropPorts);
    return success();
  }
  std::string getName() const override { return "root-port-pruner"; }
};

/// A sample reduction pattern that replaces instances of `firrtl.extmodule`
/// with wires.
struct ExtmoduleInstanceRemover : public Reduction {
  void beforeReduction(mlir::ModuleOp op) override {
    symbols.clear();
    nlaRemover.clear();
  }
  void afterReduction(mlir::ModuleOp op) override { nlaRemover.remove(op); }

  uint64_t match(Operation *op) override {
    if (auto instOp = dyn_cast<firrtl::InstanceOp>(op))
      return isa<firrtl::FExtModuleOp>(
          instOp.getReferencedModule(symbols.getNearestSymbolTable(instOp)));
    return 0;
  }
  LogicalResult rewrite(Operation *op) override {
    auto instOp = cast<firrtl::InstanceOp>(op);
    auto portInfo =
        instOp.getReferencedModule(symbols.getNearestSymbolTable(instOp))
            .getPorts();
    ImplicitLocOpBuilder builder(instOp.getLoc(), instOp);
    SmallVector<Value> replacementWires;
    for (firrtl::PortInfo info : portInfo) {
      auto wire = builder.create<firrtl::WireOp>(
          info.type, (Twine(instOp.getName()) + "_" + info.getName()).str());
      if (info.isOutput()) {
        auto inv = builder.create<firrtl::InvalidValueOp>(info.type);
        builder.create<firrtl::ConnectOp>(wire, inv);
      }
      replacementWires.push_back(wire);
    }
    nlaRemover.markNLAsInOperation(instOp);
    instOp.replaceAllUsesWith(std::move(replacementWires));
    instOp->erase();
    return success();
  }
  std::string getName() const override { return "extmodule-instance-remover"; }
  bool acceptSizeIncrease() const override { return true; }

  SymbolCache symbols;
  NLARemover nlaRemover;
};

/// A sample reduction pattern that pushes connected values through wires.
struct ConnectForwarder : public Reduction {
  void beforeReduction(mlir::ModuleOp op) override { opsToErase.clear(); }
  void afterReduction(mlir::ModuleOp op) override {
    for (auto *op : opsToErase)
      op->dropAllReferences();
    for (auto *op : opsToErase)
      op->erase();
  }

  uint64_t match(Operation *op) override {
    if (!isa<firrtl::FConnectLike>(op))
      return 0;
    auto dest = op->getOperand(0);
    auto src = op->getOperand(1);
    auto *destOp = dest.getDefiningOp();
    auto *srcOp = src.getDefiningOp();
    if (dest == src)
      return 0;

    // Ensure that the destination is something we should be able to forward
    // through.
    if (!isa_and_nonnull<firrtl::WireOp>(destOp))
      return 0;

    // Ensure that the destination is connected to only once, and all uses of
    // the connection occur after the definition of the source.
    unsigned numConnects = 0;
    for (auto &use : dest.getUses()) {
      auto *op = use.getOwner();
      if (use.getOperandNumber() == 0 && isa<firrtl::FConnectLike>(op)) {
        if (++numConnects > 1)
          return 0;
        continue;
      }
      if (srcOp && !srcOp->isBeforeInBlock(op))
        return 0;
    }

    return 1;
  }

  LogicalResult rewrite(Operation *op) override {
    auto dest = op->getOperand(0);
    dest.replaceAllUsesWith(op->getOperand(1));
    opsToErase.insert(dest.getDefiningOp());
    opsToErase.insert(op);
    return success();
  }

  std::string getName() const override { return "connect-forwarder"; }

  llvm::DenseSet<Operation *> opsToErase;
};

/// A sample reduction pattern that replaces a single-use wire and register with
/// an operand of the source value of the connection.
template <unsigned OpNum>
struct ConnectSourceOperandForwarder : public Reduction {
  uint64_t match(Operation *op) override {
    if (!isa<firrtl::ConnectOp, firrtl::StrictConnectOp>(op))
      return 0;
    auto dest = op->getOperand(0);
    auto *destOp = dest.getDefiningOp();

    // Ensure that the destination is used only once.
    if (!destOp || !destOp->hasOneUse() ||
        !isa<firrtl::WireOp, firrtl::RegOp, firrtl::RegResetOp>(destOp))
      return 0;

    auto *srcOp = op->getOperand(1).getDefiningOp();
    if (!srcOp || OpNum >= srcOp->getNumOperands())
      return 0;

    auto resultTy = dest.getType().dyn_cast<firrtl::FIRRTLBaseType>();
    auto opTy =
        srcOp->getOperand(OpNum).getType().dyn_cast<firrtl::FIRRTLBaseType>();

    return resultTy && opTy &&
           resultTy.getWidthlessType() == opTy.getWidthlessType() &&
           ((resultTy.getBitWidthOrSentinel() == -1) ==
            (opTy.getBitWidthOrSentinel() == -1)) &&
           resultTy.isa<firrtl::UIntType, firrtl::SIntType>();
  }

  LogicalResult rewrite(Operation *op) override {
    auto *destOp = op->getOperand(0).getDefiningOp();
    auto *srcOp = op->getOperand(1).getDefiningOp();
    auto forwardedOperand = srcOp->getOperand(OpNum);
    ImplicitLocOpBuilder builder(destOp->getLoc(), destOp);
    Value newDest;
    if (auto wire = dyn_cast<firrtl::WireOp>(destOp))
      newDest = builder.create<firrtl::WireOp>(forwardedOperand.getType(),
                                               wire.getName());
    else {
      auto regName = destOp->getAttrOfType<StringAttr>("name");
      // We can promote the register into a wire but we wouldn't do here because
      // the error might be caused by the register.
      auto clock = destOp->getOperand(0);
      newDest = builder.create<firrtl::RegOp>(forwardedOperand.getType(), clock,
                                              regName ? regName.str() : "");
    }

    // Create new connection between a new wire and the forwarded operand.
    builder.setInsertionPointAfter(op);
    if (isa<firrtl::ConnectOp>(op))
      builder.create<firrtl::ConnectOp>(newDest, forwardedOperand);
    else
      builder.create<firrtl::StrictConnectOp>(newDest, forwardedOperand);

    // Remove the old connection and destination. We don't have to replace them
    // because destination has only one use.
    op->erase();
    destOp->erase();
    pruneUnusedOps(srcOp);

    return success();
  }
  std::string getName() const override {
    return ("connect-source-operand-" + Twine(OpNum) + "-forwarder").str();
  }
};

/// A sample reduction pattern that tries to remove aggregate wires by replacing
/// all subaccesses with new independent wires. This can disentangle large
/// unused wires that are otherwise difficult to collect due to the subaccesses.
struct DetachSubaccesses : public Reduction {
  void beforeReduction(mlir::ModuleOp op) override { opsToErase.clear(); }
  void afterReduction(mlir::ModuleOp op) override {
    for (auto *op : opsToErase)
      op->dropAllReferences();
    for (auto *op : opsToErase)
      op->erase();
  }
  uint64_t match(Operation *op) override {
    // Only applies to wires and registers that are purely used in subaccess
    // operations.
    return isa<firrtl::WireOp, firrtl::RegOp, firrtl::RegResetOp>(op) &&
           llvm::all_of(op->getUses(), [](auto &use) {
             return use.getOperandNumber() == 0 &&
                    isa<firrtl::SubfieldOp, firrtl::SubindexOp,
                        firrtl::SubaccessOp>(use.getOwner());
           });
  }
  LogicalResult rewrite(Operation *op) override {
    assert(match(op));
    OpBuilder builder(op);
    bool isWire = isa<firrtl::WireOp>(op);
    Value invalidClock;
    if (!isWire)
      invalidClock = builder.create<firrtl::InvalidValueOp>(
          op->getLoc(), firrtl::ClockType::get(op->getContext()));
    for (Operation *user : llvm::make_early_inc_range(op->getUsers())) {
      builder.setInsertionPoint(user);
      auto type = user->getResult(0).getType();
      Operation *replOp;
      if (isWire)
        replOp = builder.create<firrtl::WireOp>(user->getLoc(), type);
      else
        replOp =
            builder.create<firrtl::RegOp>(user->getLoc(), type, invalidClock);
      user->replaceAllUsesWith(replOp);
      opsToErase.insert(user);
    }
    opsToErase.insert(op);
    return success();
  }
  std::string getName() const override { return "detach-subaccesses"; }
  llvm::DenseSet<Operation *> opsToErase;
};

/// This reduction removes symbols on node ops. Name preservation creates a lot
/// of nodes ops with symbols to keep name information but it also prevents
/// normal canonicalizations.
struct NodeSymbolRemover : public Reduction {

  uint64_t match(Operation *op) override {
    if (auto nodeOp = dyn_cast<firrtl::NodeOp>(op))
      return nodeOp.getInnerSym() &&
             !nodeOp.getInnerSym()->getSymName().getValue().empty();
    return 0;
  }

  LogicalResult rewrite(Operation *op) override {
    auto nodeOp = cast<firrtl::NodeOp>(op);
    nodeOp.removeInnerSymAttr();
    return success();
  }

  std::string getName() const override { return "node-symbol-remover"; }
};

/// A sample reduction pattern that eagerly inlines instances.
struct EagerInliner : public Reduction {
  void beforeReduction(mlir::ModuleOp op) override {
    symbols.clear();
    nlaRemover.clear();
  }
  void afterReduction(mlir::ModuleOp op) override { nlaRemover.remove(op); }

  uint64_t match(Operation *op) override {
    auto instOp = dyn_cast<firrtl::InstanceOp>(op);
    if (!instOp)
      return 0;
    auto tableOp = SymbolTable::getNearestSymbolTable(instOp);
    auto moduleOp = instOp.getReferencedModule(symbols.getSymbolTable(tableOp));
    if (!isa<firrtl::FModuleOp>(moduleOp.getOperation()))
      return 0;
    return symbols.getSymbolUserMap(tableOp).getUsers(moduleOp).size() == 1;
  }

  LogicalResult rewrite(Operation *op) override {
    auto instOp = cast<firrtl::InstanceOp>(op);
    LLVM_DEBUG(llvm::dbgs()
               << "Inlining instance `" << instOp.getName() << "`\n");
    SmallVector<Value> argReplacements;
    ImplicitLocOpBuilder builder(instOp.getLoc(), instOp);
    for (unsigned i = 0, e = instOp.getNumResults(); i != e; ++i) {
      auto result = instOp.getResult(i);
      auto name = builder.getStringAttr(Twine(instOp.getName()) + "_" +
                                        instOp.getPortNameStr(i));
      auto wire = builder.create<firrtl::WireOp>(
          result.getType(), name, firrtl::NameKindEnum::DroppableName,
          instOp.getPortAnnotation(i), StringAttr{});
      result.replaceAllUsesWith(wire);
      argReplacements.push_back(wire);
    }
    auto tableOp = SymbolTable::getNearestSymbolTable(instOp);
    auto moduleOp = cast<firrtl::FModuleOp>(
        instOp.getReferencedModule(symbols.getSymbolTable(tableOp))
            .getOperation());
    for (auto &op : llvm::make_early_inc_range(*moduleOp.getBodyBlock())) {
      op.remove();
      builder.insert(&op);
      for (auto &operand : op.getOpOperands())
        if (auto blockArg = operand.get().dyn_cast<BlockArgument>())
          operand.set(argReplacements[blockArg.getArgNumber()]);
    }
    nlaRemover.markNLAsInOperation(instOp);
    instOp->erase();
    moduleOp->erase();
    return success();
  }

  std::string getName() const override { return "eager-inliner"; }
  bool acceptSizeIncrease() const override { return true; }

  SymbolCache symbols;
  NLARemover nlaRemover;
};

//===----------------------------------------------------------------------===//
// Reduction Registration
//===----------------------------------------------------------------------===//

static std::unique_ptr<Pass> createSimpleCanonicalizerPass() {
  GreedyRewriteConfig config;
  config.useTopDownTraversal = true;
  config.enableRegionSimplification = false;
  return createCanonicalizerPass(config);
}

void circt::createAllReductions(
    MLIRContext *context,
    llvm::function_ref<void(std::unique_ptr<Reduction>)> add) {
  // Gather a list of reduction patterns that we should try. Ideally these are
  // sorted by decreasing reduction potential/benefit. For example, things that
  // can knock out entire modules while being cheap should be tried first,
  // before trying to tweak operands of individual arithmetic ops.
  add(std::make_unique<PassReduction>(context, firrtl::createLowerCHIRRTLPass(),
                                      true, true));
  add(std::make_unique<PassReduction>(context, firrtl::createInferWidthsPass(),
                                      true, true));
  add(std::make_unique<PassReduction>(context, firrtl::createInferResetsPass(),
                                      true, true));
  add(std::make_unique<ModuleExternalizer>());
  add(std::make_unique<InstanceStubber>());
  add(std::make_unique<MemoryStubber>());
  add(std::make_unique<EagerInliner>());
  add(std::make_unique<PassReduction>(
      context, firrtl::createLowerFIRRTLTypesPass(), true, true));
  add(std::make_unique<PassReduction>(context, firrtl::createExpandWhensPass(),
                                      true, true));
  add(std::make_unique<PassReduction>(context, firrtl::createInlinerPass()));
  add(std::make_unique<PassReduction>(context,
                                      createSimpleCanonicalizerPass()));
  add(std::make_unique<PassReduction>(context,
                                      firrtl::createIMConstPropPass()));
  add(std::make_unique<PassReduction>(
      context, firrtl::createRemoveUnusedPortsPass(/*ignoreDontTouch=*/true)));
  add(std::make_unique<PassReduction>(context, createCSEPass()));
  add(std::make_unique<NodeSymbolRemover>());
  add(std::make_unique<ConnectForwarder>());
  add(std::make_unique<ConnectInvalidator>());
  add(std::make_unique<Constantifier>());
  add(std::make_unique<OperandForwarder<0>>());
  add(std::make_unique<OperandForwarder<1>>());
  add(std::make_unique<OperandForwarder<2>>());
  add(std::make_unique<OperationPruner>());
  add(std::make_unique<DetachSubaccesses>());
  add(std::make_unique<AnnotationRemover>());
  add(std::make_unique<RootPortPruner>());
  add(std::make_unique<ExtmoduleInstanceRemover>());
  add(std::make_unique<ConnectSourceOperandForwarder<0>>());
  add(std::make_unique<ConnectSourceOperandForwarder<1>>());
  add(std::make_unique<ConnectSourceOperandForwarder<2>>());
}
