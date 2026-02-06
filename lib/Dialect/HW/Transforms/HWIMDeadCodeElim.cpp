//===- IMDeadCodeElim.cpp - Intermodule Dead Code Elimination ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWModuleGraph.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/HW/InnerSymbolTable.h"
#include "circt/Support/Debug.h"
#include "circt/Support/InstanceGraph.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMapInfoVariant.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/InstructionNamer.h"
#include <variant>
#include <vector>

#define DEBUG_TYPE "hw-imdeadcodeelim"

namespace circt {
namespace hw {
#define GEN_PASS_DEF_IMDEADCODEELIM
#include "circt/Dialect/HW/Passes.h.inc"
} // namespace hw
} // namespace circt

using namespace circt;
using namespace hw;

namespace {
struct HWIMDeadCodeElim
    : circt::hw::impl::IMDeadCodeElimBase<HWIMDeadCodeElim> {
  using Base::Base;

  void runOnOperation() override;

private:
  using ElementType = std::variant<Value, HWModuleLike, InstanceOp, HierPathOp>;

  void markAlive(ElementType element) {
    if (!liveElements.insert(element).second)
      return;
    worklist.push_back(element);
  }

  bool isKnownAlive(InstanceOp instance) const {
    return liveElements.count(instance);
  }
  bool isKnownAlive(Value value) const {
    assert(value && "null should not be used");
    return liveElements.count(value);
  }
  bool isAssumedDead(Value value) const { return !isKnownAlive(value); }
  bool isAssumedDead(Operation *op) const {
    return llvm::none_of(op->getResults(),
                         [&](Value value) { return isKnownAlive(value); });
  }
  bool isBlockExecutable(Block *block) { return executableBlocks.count(block); }

  SmallVector<ElementType, 64> worklist;
  llvm::DenseSet<ElementType> liveElements;

  circt::hw::InnerRefNamespace *innerRefNamespace;
  mlir::SymbolTable *symbolTable;
  InstanceGraph *instanceGraph;
  DenseMap<InstanceOp, SmallVector<HierPathOp>> instanceToHierPaths;
  DenseSet<Block *> executableBlocks;

  void markBlockExecutable(Block *block);

  void visitInstanceOp(InstanceOp instance);
  void visitHierPathOp(HierPathOp hierpath);
  void visitValue(Value value);

  void rewriteModuleSignature(HWModuleOp module);
  void eraseEmptyModule(HWModuleOp module);
  void rewriteModuleBody(HWModuleOp module);

  void markUnknownSideEffectOp(Operation *op);
  void markInstanceOp(InstanceOp instance);

  void markBlockUndeletable(Operation *op) {
    markAlive(op->getParentOfType<HWModuleOp>());
  }
};
} // namespace

static bool hasUnknownSideEffect(Operation *op) {
  if (!(mlir::isMemoryEffectFree(op) ||
        mlir::hasSingleEffect<mlir::MemoryEffects::Allocate>(op) ||
        mlir::hasSingleEffect<mlir::MemoryEffects::Read>(op))) {
    return true;
  }
  if (auto innerSymOp = dyn_cast<InnerSymbolOpInterface>(op)) {
    if (innerSymOp.getInnerName().has_value())
      return true;
  }

  return false;
}

void HWIMDeadCodeElim::markUnknownSideEffectOp(Operation *op) {
  // For operations with side effects, pessimistically mark results and
  // operands as alive.
  for (auto result : op->getResults())
    markAlive(result);
  for (auto operand : op->getOperands())
    markAlive(operand);
  markBlockUndeletable(op);
}

void HWIMDeadCodeElim::markInstanceOp(InstanceOp instance) {
  // Get the module being referenced.
  auto moduleNameAttr = instance.getModuleNameAttr().getAttr();
  auto *node = instanceGraph->lookup(moduleNameAttr);
  if (!node)
    return;
  auto op = node->getModule();

  if (instance.getInnerSym().has_value()) {
    markAlive(instance);
    if (auto module = dyn_cast<HWModuleOp>(*op))
      markBlockExecutable(module.getBodyBlock());
  }

  // If this is an extmodule, just remember that any inputs and inouts are
  // alive.
  // Inputs are exactly what are passed into the module.
  if (!isa<HWModuleOp>(op)) {
    for (auto operand : instance->getOperands()) {
      markAlive(operand);
    }
    markAlive(instance);
    return;
  }

  // Otherwise this is a defined module.
  auto fModule = cast<HWModuleOp>(op);
  markBlockExecutable(fModule.getBodyBlock());
}

void HWIMDeadCodeElim::markBlockExecutable(Block *block) {
  if (!executableBlocks.insert(block).second)
    return; // Already executable.

  auto hwmodule = dyn_cast<HWModuleOp>(block->getParentOp());
  if (hwmodule && hwmodule.isPublic())
    markAlive(hwmodule);

  for (auto &op : *block) {
    if (auto instance = dyn_cast<InstanceOp>(op))
      markInstanceOp(instance);
    else if (hasUnknownSideEffect(&op)) {
      markUnknownSideEffectOp(&op);
      // Recursively mark any blocks contained within these operations as
      // executable.
      for (auto &region : op.getRegions())
        for (auto &block : region.getBlocks())
          markBlockExecutable(&block);
    }
  }
}

void HWIMDeadCodeElim::visitInstanceOp(InstanceOp instance) {

  auto moduleName = instance.getReferencedModuleNameAttr();
  auto *moduleNode = instanceGraph->lookup(moduleName);
  if (auto module = dyn_cast<HWModuleOp>(moduleNode->getModule())) {

    if (instance.getInnerSym().has_value()) {
      markAlive(module);
    }

    // Propagate liveness through hierpath.
    for (auto hierPath : instanceToHierPaths[instance])
      markAlive(hierPath);

    // All block args are inputs
    for (auto &blockArg : module.getBody().getArguments()) {
      auto portIndex = blockArg.getArgNumber();

      if (isKnownAlive(blockArg))
        markAlive(instance->getOperand(portIndex));
    }
  }

  if (auto extModule = dyn_cast<HWModuleExternOp>(moduleNode->getModule())) {
    markAlive(extModule);
  }
}

void HWIMDeadCodeElim::visitHierPathOp(hw::HierPathOp hierPathOp) {
  // If the hierpath is alive, mark all instances on the path alive.
  for (auto path : hierPathOp.getNamepathAttr()) {
    if (auto innerRef = dyn_cast<hw::InnerRefAttr>(path)) {
      auto *op = innerRefNamespace->lookupOp(innerRef);

      if (auto instance = dyn_cast_or_null<InstanceOp>(op)) {
        markAlive(instance);
      }
    }
  }
}

/// Propagate liveness through \p value.
/// If value is a block arg, it means that it had been found
/// live through propagating within a module, so we mark corresponding
/// instance inputs as alive.
/// If it is the result of an instance op, we mark the corresponding
/// value in the body of the module as alive.
/// Otherwise, we mark all operands of the operation defining `value`
/// as alive.
void HWIMDeadCodeElim::visitValue(Value value) {
  assert(isKnownAlive(value) && "only alive values reach here");

  // Requiring an input port propagates the liveness to each instance.
  // All arg block elements are inputs
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    if (auto module =
            dyn_cast<HWModuleOp>(blockArg.getParentBlock()->getParentOp())) {

      for (auto *instRec : instanceGraph->lookup(module)->uses()) {
        if (!instRec->getInstance())
          continue;

        if (auto instance = dyn_cast<InstanceOp>(instRec->getInstance())) {
          if (liveElements.contains(instance))
            markAlive(instance->getOperand(blockArg.getArgNumber()));
        }
      }
    }
  }

  // Marking an instance port as alive propagates to the corresponding port of
  // the module.
  if (auto instance = value.getDefiningOp<InstanceOp>()) {
    auto instanceResult = cast<mlir::OpResult>(value);
    // Update the src, when it's an instance op.
    auto moduleName = instance.getReferencedModuleNameAttr();
    auto *node = instanceGraph->lookupOrNull(moduleName);

    if (!node)
      return;

    Operation *moduleOp = node->getModule();
    auto module = dyn_cast_or_null<HWModuleOp>(moduleOp);

    // Propagate liveness only when a port is output.
    if (!module)
      return;

    markAlive(instance);

    auto *moduleOutputOp = module.getBodyBlock()->getTerminator();
    auto modulePortVal =
        moduleOutputOp->getOperand(instanceResult.getResultNumber());
    markAlive(modulePortVal);

    visitInstanceOp(instance);
    return;
  }

  // If the value is defined by an operation, mark its operands alive and any
  // nested blocks executable.
  if (auto *op = value.getDefiningOp()) {
    for (auto operand : op->getOperands())
      markAlive(operand);
    for (auto &region : op->getRegions())
      for (auto &block : region)
        markBlockExecutable(&block);
  }
}

void HWIMDeadCodeElim::runOnOperation() {

  instanceGraph = &getAnalysis<hw::InstanceGraph>();
  symbolTable = &getAnalysis<SymbolTable>();
  auto &istc = getAnalysis<hw::InnerSymbolTableCollection>();

  circt::hw::InnerRefNamespace theInnerRefNamespace{*symbolTable, istc};
  innerRefNamespace = &theInnerRefNamespace;

  for (auto module : getOperation().getOps<hw::HWModuleOp>()) {
    // Mark the ports of public modules as alive.
    if (module.isPublic()) {
      markBlockExecutable(module.getBodyBlock());

      // Do not mark inputs as alive, since they are only
      // internally relevant.

      // Mark all output values (i.e. SSA vals passed to hw.output) as alive
      auto *moduleOutputOp = module.getBodyBlock()->getTerminator();
      for (auto port : moduleOutputOp->getOperands()) {
        markAlive(port);
      }
    }
  }

  // If an element changed liveness then propagate liveness through it.
  while (!worklist.empty()) {
    auto v = worklist.pop_back_val();
    if (auto *value = std::get_if<Value>(&v)) {
      visitValue(*value);
    } else if (auto *instance = std::get_if<InstanceOp>(&v)) {
      visitInstanceOp(*instance);
    } else if (auto *hierpath = std::get_if<hw::HierPathOp>(&v)) {
      visitHierPathOp(*hierpath);
    } else if (auto *moduleLike = std::get_if<HWModuleLike>(&v)) {
      // TODO: Maybe some processing could be done in ModuleOp's?
      continue;
    }
  }

  if (printLiveness) {
    auto liveAttr = StringAttr::get(&getContext(), "LIVE");
    auto deadAttr = StringAttr::get(&getContext(), "DEAD");

    auto setLiveness = [&](Operation *op,
                           SmallVector<mlir::Attribute> &resultLiveness) {
      op->setAttr("val-liveness",
                  ArrayAttr::get(&getContext(), resultLiveness));
    };

    getOperation()->walk([&](Operation *op) {
      SmallVector<mlir::Attribute> resultLiveness;

      if (auto module = dyn_cast<HWModuleOp>(op)) {
        for (auto result : module.getBodyBlock()->getArguments()) {
          resultLiveness.push_back(isKnownAlive(result) ? liveAttr : deadAttr);
        }
        setLiveness(op, resultLiveness);
        module->setAttr("op-liveness", isBlockExecutable(module.getBodyBlock())
                                           ? liveAttr
                                           : deadAttr);
        return;
      }

      if (auto outputOp = dyn_cast<OutputOp>(op)) {
        for (auto operand : outputOp->getOperands()) {
          resultLiveness.push_back(isKnownAlive(operand) ? liveAttr : deadAttr);
        }
        setLiveness(op, resultLiveness);
        return;
      }

      if (op->getNumResults()) {
        for (auto result : op->getResults()) {
          resultLiveness.push_back(isKnownAlive(result) ? liveAttr : deadAttr);
        }
        setLiveness(op, resultLiveness);

        if (auto instance = dyn_cast<InstanceOp>(op)) {
          instance->setAttr("op-liveness",
                            isKnownAlive(instance) ? liveAttr : deadAttr);
        } else {
          op->setAttr("op-liveness", isAssumedDead(op) ? deadAttr : liveAttr);
        }

        return;
      }

      return;
    });
  }

  // Clean up data structures.
  executableBlocks.clear();
  liveElements.clear();
  instanceToHierPaths.clear();
}
