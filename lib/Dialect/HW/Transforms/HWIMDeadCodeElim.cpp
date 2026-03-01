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
#include "llvm/ADT/DenseMapInfoVariant.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <variant>

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
  using ElementType = std::variant<Value, HWModuleLike, HWInstanceLike>;

  void markAlive(ElementType element) {
    if (!liveElements.insert(element).second)
      return;
    worklist.push_back(element);
  }

  bool isKnownAlive(HWInstanceLike instanceLike) const {
    return liveElements.count(instanceLike);
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

  InstanceGraph *instanceGraph;
  DenseSet<Block *> executableBlocks;

  void markBlockExecutable(Block *block);

  void visitInstanceLike(HWInstanceLike instanceLike);
  void visitValue(Value value);

  void markUnknownSideEffectOp(Operation *op);
  void markInstanceLike(HWInstanceLike instanceLike);

  void markBlockUndeletable(Operation *op) {

    markAlive(op->getParentOfType<HWModuleLike>());
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

void HWIMDeadCodeElim::markInstanceLike(HWInstanceLike instanceLike) {

  auto moduleNames = instanceLike.getReferencedModuleNames();
  for (auto moduleName : moduleNames) {
    auto moduleNameAttr = mlir::StringAttr::get(&getContext(), moduleName);
    auto *node = instanceGraph->lookup(moduleNameAttr);

    if (!node)
      continue;

    auto op = node->getModule();

    // If this is an extmodule, just remember that any inputs and inouts are
    // alive.
    // Inputs are exactly what are passed into the module.
    if (!dyn_cast<HWModuleOp>(op.getOperation())) {
      for (auto operand : instanceLike->getOperands())
        markAlive(operand);

      markAlive(instanceLike);
    }

    if (auto moduleLike = dyn_cast<HWModuleLike>(op.getOperation()))
      markBlockExecutable(moduleLike.getBodyBlock());
  }
}

void HWIMDeadCodeElim::markBlockExecutable(Block *block) {
  if (!block)
    return;

  if (!executableBlocks.insert(block).second)
    return; // Already executable.

  auto moduleLike = dyn_cast<HWModuleLike>(block->getParentOp());
  // The only case when we do not mark the module alive automatically,
  // is if it is a private HW module
  auto module = dyn_cast<HWModuleOp>(moduleLike.getOperation());
  if (!module)
    markAlive(moduleLike);
  else {
    if (module.isPublic())
      markAlive(module);
  }

  for (auto &op : *block) {
    if (auto instance = dyn_cast<HWInstanceLike>(op))
      markInstanceLike(instance);
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
/// Propagate liveness backwards through a HWInstanceLike op.
/// For all possible ModuleLike's that is being instantiated,
/// - If it is a HWModuleOp, mark outputs selectively based on
///   their liveness in the module,
/// - If it is a ModuleLike, but not a HWModuleOp, mark all inputs
///   as alive.
///   TODO: this might not be the best idea, since we can still trace
///         out if something is actually used in a non HWModuleOp.
void HWIMDeadCodeElim::visitInstanceLike(HWInstanceLike instanceLike) {

  auto moduleNames = instanceLike.getReferencedModuleNames();
  for (StringRef moduleName : moduleNames) {
    mlir::StringAttr moduleNameAttr =
        mlir::StringAttr::get(&getContext(), moduleName);

    auto *moduleNode = instanceGraph->lookup(moduleNameAttr);

    if (!moduleNode)
      continue;

    // We apply exact argument liveness iff the module is HW native.
    if (auto module =
            dyn_cast<HWModuleOp>(moduleNode->getModule().getOperation())) {

      // All block args are inputs
      for (auto blockArg : module.getBodyBlock()->getArguments()) {
        auto portIndex = blockArg.getArgNumber();

        if (isKnownAlive(blockArg))
          markAlive(instanceLike->getOperand(portIndex));
      }

      continue;
    }

    // Otherwise we mark all outputs as alive
    if (auto extModule =
            dyn_cast<HWModuleLike>(moduleNode->getModule().getOperation()))
      markAlive(extModule);
  }
}

/// Propagate liveness through \p value.
/// - If value is a block arg, it means that it had been found
///   live through propagating within a module, so we mark corresponding
///   instance inputs as alive.
/// - If it is the result of an instance op, we mark the corresponding
///   value in the body of the module as alive.
/// - Otherwise, we mark all operands of the operation defining `value`
///   as alive.
void HWIMDeadCodeElim::visitValue(Value value) {
  assert(isKnownAlive(value) && "only alive values reach here");

  // Requiring an input port propagates the liveness to each instance.
  // All arg block elements are inputs
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    if (auto module =
            dyn_cast<HWModuleLike>(blockArg.getParentBlock()->getParentOp())) {

      for (auto *instRec : instanceGraph->lookup(module)->uses()) {
        if (!instRec->getInstance())
          continue;

        if (auto instance = dyn_cast<HWInstanceLike>(
                instRec->getInstance().getOperation())) {
          if (liveElements.contains(instance))
            markAlive(instance->getOperand(blockArg.getArgNumber()));
        }
      }
    }
  }

  // Marking an instance port as alive propagates to the corresponding port of
  // the module.
  if (auto instanceLike = value.getDefiningOp<HWInstanceLike>()) {
    auto instanceResult = cast<mlir::OpResult>(value);
    // Update the src, when it's an instance op.

    // For each module that the instance could refer to,
    // mark
    auto moduleNames = instanceLike.getReferencedModuleNames();
    for (StringRef moduleName : moduleNames) {
      mlir::StringAttr moduleNameAttr =
          mlir::StringAttr::get(&getContext(), moduleName);
      auto *node = instanceGraph->lookupOrNull(moduleNameAttr);

      if (!node)
        continue;

      Operation *moduleOp = node->getModule().getOperation();
      auto moduleLike = dyn_cast<HWModuleLike>(moduleOp);

      if (!moduleLike)
        continue;

      if (!moduleLike.getBodyBlock())
        continue;
      auto *moduleOutputOp = moduleLike.getBodyBlock()->getTerminator();
      auto modulePortVal =
          moduleOutputOp->getOperand(instanceResult.getResultNumber());
      markAlive(modulePortVal);
      markAlive(instanceLike);
    }

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

  for (auto moduleLike : getOperation().getOps<hw::HWModuleLike>()) {
    // Any ModuleLike that is not a private module will be marked
    // executable, and all its output ports alive.
    if (auto module = dyn_cast<HWModuleOp>(moduleLike.getOperation())) {
      if (!module.isPublic())
        continue;
    }

    if (!moduleLike.getBodyBlock())
      continue;

    markBlockExecutable(moduleLike.getBodyBlock());

    // Do not mark inputs as alive, since they are only
    // internally relevant.

    // Mark all output values (i.e. SSA vals passed to hw.output) as alive
    if (moduleLike.getBodyBlock() == nullptr ||
        moduleLike.getBodyBlock()->empty())
      continue;
    auto *moduleOutputOp = moduleLike.getBodyBlock()->getTerminator();
    if (!dyn_cast<OutputOp>(moduleOutputOp))
      continue;
    for (auto port : moduleOutputOp->getOperands())
      markAlive(port);
  }

  // If an element changed liveness then propagate liveness through it.
  while (!worklist.empty()) {
    auto v = worklist.pop_back_val();
    if (auto *value = std::get_if<Value>(&v)) {
      visitValue(*value);
    } else if (auto *instance = std::get_if<HWInstanceLike>(&v)) {
      visitInstanceLike(*instance);
    } else if (auto *moduleLike = std::get_if<HWModuleLike>(&v)) {
      continue;
    }
  }

  if (printLiveness) {
    auto liveAttr = StringAttr::get(&getContext(), "LIVE");
    auto deadAttr = StringAttr::get(&getContext(), "DEAD");

    auto getLiveness = [&](bool alive) { return alive ? liveAttr : deadAttr; };

    auto getValLiveness = [&](ValueRange values) {
      SmallVector<Attribute> liveness;
      for (auto value : values)
        liveness.push_back(getLiveness(isKnownAlive(value)));
      return ArrayAttr::get(&getContext(), liveness);
    };

    getOperation()->walk([&](Operation *op) {
      if (auto module = dyn_cast<HWModuleLike>(op)) {
        if (!module.getBodyBlock())
          return;

        op->setAttr("op-liveness",
                    getLiveness(isBlockExecutable(module.getBodyBlock())));
        op->setAttr("val-liveness",
                    getValLiveness(module.getBodyBlock()->getArguments()));
        return;
      }

      if (auto outputOp = dyn_cast<OutputOp>(op)) {
        op->setAttr("val-liveness", getValLiveness(outputOp->getOperands()));
        return;
      }

      if (op->getNumResults()) {
        if (auto instance = dyn_cast<HWInstanceLike>(op))
          op->setAttr("op-liveness", getLiveness(isKnownAlive(instance)));
        else
          op->setAttr("op-liveness", getLiveness(!isAssumedDead(op)));
        op->setAttr("val-liveness", getValLiveness(op->getResults()));
      }
    });
  }

  // Clean up data structures.
  executableBlocks.clear();
  liveElements.clear();
}
