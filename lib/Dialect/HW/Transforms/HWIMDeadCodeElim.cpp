//===- IMDeadCodeElim.cpp - Intermodule Dead Code Elimination ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Support/Utils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMapInfoVariant.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallVector.h"
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

  void rewriteModuleSignature(HWModuleOp module);
  void eraseEmptyModule(HWModuleOp module);
  void rewriteModuleBody(HWModuleOp module);

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

/// Clone \p instance, but with ports deleted according to
/// the \p inErasures and \p outErasures BitVectors.
static InstanceOp cloneWithErasedPorts(InstanceOp &instance,
                                       const llvm::BitVector &inErasures,
                                       const llvm::BitVector &outErasures) {
  assert(outErasures.size() >= instance->getNumResults() &&
         "out_erasures is not at least as large as getNumResults()");
  assert(inErasures.size() >= instance->getNumOperands() &&
         "in_erasures is not at least as large as getNumOperands()");

  // Restrict outputs
  SmallVector<Type> newResultTypes = removeElementsAtIndices<Type>(
      SmallVector<Type>(instance->result_type_begin(),
                        instance->result_type_end()),
      outErasures);
  auto newResultNames = removeElementsAtIndices<mlir::Attribute>(
      SmallVector<mlir::Attribute>(instance.getResultNames().begin(),
                                   instance.getResultNames().end()),
      outErasures);

  // Restrict inputs
  auto newOperands = removeElementsAtIndices<mlir::Value>(
      SmallVector<mlir::Value>(instance->getOperands().begin(),
                               instance->getOperands().end()),
      inErasures);
  auto newOperandNames = removeElementsAtIndices<mlir::Attribute>(
      SmallVector<mlir::Attribute>(instance.getArgNames().begin(),
                                   instance.getArgNames().end()),
      inErasures);

  ImplicitLocOpBuilder builder(instance->getLoc(), instance);

  auto newOpNamesArrayAttr = builder.getArrayAttr(newOperandNames);
  auto newResultNamesArrayAttr = builder.getArrayAttr(newResultNames);

  auto newInstance = InstanceOp::create(
      builder, instance->getLoc(), newResultTypes,
      instance.getInstanceNameAttr(), instance.getModuleName(), newOperands,
      newOpNamesArrayAttr, newResultNamesArrayAttr, instance.getParameters(),
      instance.getInnerSymAttr());

  return newInstance;
}

/// Static method for cloning \p instance of type InstanceOp
/// with in- and output ports erased based on \p inErasures
/// and \p outErasures respectively. The users of the results
/// of the instance are also updated.
static InstanceOp
cloneWithErasePortsAndReplaceUses(InstanceOp &instance,
                                  const llvm::BitVector &inErasures,
                                  const llvm::BitVector &outErasures) {

  auto newInstance = cloneWithErasedPorts(instance, inErasures, outErasures);

  // Replace all input operands
  size_t erased = 0;
  for (size_t index = 0, e = instance->getNumResults(); index < e; ++index) {
    auto r1 = instance->getResult(index);
    if (outErasures[index]) {
      ++erased;
      continue;
    }
    auto r2 = newInstance->getResult(index - erased);
    r1.replaceAllUsesWith(r2);
  }

  return newInstance;
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

  for (StringAttr moduleName :
       instanceLike.getReferencedModuleNamesAttr().getAsRange<StringAttr>()) {
    auto *node = instanceGraph->lookup(moduleName);

    if (!node)
      continue;

    auto op = node->getModule();

    // If this is an extmodule, just remember that any inputs and inouts are
    // alive.
    // Inputs are exactly what are passed into the module.
    if (!isa<HWModuleOp>(op.getOperation())) {
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
  else if (module.isPublic())
    markAlive(module);

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

  for (mlir::StringAttr moduleName :
       instanceLike.getReferencedModuleNamesAttr().getAsRange<StringAttr>()) {
    auto *moduleNode = instanceGraph->lookup(moduleName);

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
    for (mlir::StringAttr moduleName :
         instanceLike.getReferencedModuleNamesAttr().getAsRange<StringAttr>()) {
      auto *node = instanceGraph->lookupOrNull(moduleName);

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
    if (moduleLike.getBodyBlock()->empty())
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
    } else if (std::get_if<HWModuleLike>(&v)) {
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

    return;
  }

  // Rewrite module signatures. Non-executable modules are still rewritten
  // (dead ports stripped, dead instances removed) but are never erased, to
  // avoid invalidating symbol references (e.g. sv.verbatim "{{0}}" {symbols =
  // [@mod]}).
  for (auto module :
       llvm::make_early_inc_range(getOperation().getOps<HWModuleOp>()))
    rewriteModuleSignature(module);

  for (auto module :
       llvm::make_early_inc_range(getOperation().getOps<HWModuleOp>()))
    rewriteModuleBody(module);

  for (auto module :
       llvm::make_early_inc_range(getOperation().getOps<HWModuleOp>())) {
    eraseEmptyModule(module);
  }

  // Clean up data structures.
  executableBlocks.clear();
  liveElements.clear();
}

void HWIMDeadCodeElim::rewriteModuleSignature(HWModuleOp module) {

  igraph::InstanceGraphNode *instanceGraphNode = instanceGraph->lookup(module);
  LLVM_DEBUG(llvm::dbgs() << "Prune ports of module: " << module.getName()
                          << "\n");

  auto replaceInstanceResultWithConst =
      [&](ImplicitLocOpBuilder &builder, unsigned index, InstanceOp instance) {
        auto result = instance.getResult(index);
        assert(isAssumedDead(result));

        auto dummy =
            mlir::UnrealizedConversionCastOp::create(
                builder, ArrayRef<Type>{result.getType()}, ArrayRef<Value>{})
                ->getResult(0);
        result.replaceAllUsesWith(dummy);
        return;
      };

  // First, delete dead instances.
  for (auto *use : llvm::make_early_inc_range(instanceGraphNode->uses())) {
    auto maybeInst = use->getInstance();
    if (!maybeInst)
      continue;

    auto instance = cast<InstanceOp>(maybeInst);

    if (!isKnownAlive(instance)) {
      // Replace old instance results with dummy wires.
      ImplicitLocOpBuilder builder(instance.getLoc(), instance);
      for (auto index : llvm::seq(0u, instance.getNumResults()))
        replaceInstanceResultWithConst(builder, index, instance);
      // Make sure that we update the instance graph.
      use->erase();
      instance.erase();
    }
  }

  // Ports of public modules cannot be modified.
  if (module.isPublic())
    return;

  // Otherwise prepare data structures for tracking dead ports.
  auto *outputOp = module.getBodyBlock()->getTerminator();

  auto numInPorts = module.getBody().getNumArguments();
  auto numOutPorts = outputOp->getNumOperands();

  llvm::BitVector deadInPortBitVec(numInPorts);
  llvm::BitVector deadOutPortBitVec(numOutPorts);

  ImplicitLocOpBuilder builder(module.getLoc(), module.getContext());
  builder.setInsertionPointToStart(module.getBodyBlock());

  for (auto index : llvm::seq(0u, numInPorts)) {
    auto inPort = module.getBodyBlock()->getArgument(index);
    if (isKnownAlive(inPort))
      continue;

    auto placeholder =
        mlir::UnrealizedConversionCastOp::create(
            builder, ArrayRef<Type>{inPort.getType()}, ArrayRef<Value>{})
            ->getResult(0);
    inPort.replaceAllUsesWith(placeholder);
    deadInPortBitVec.set(index);
  }

  // Find all unused results
  unsigned erasures = 0;
  for (auto index : llvm::seq(0u, numOutPorts)) {
    auto argument = outputOp->getOperand(index - erasures);

    if (isKnownAlive(argument))
      continue;

    outputOp->eraseOperand(index - erasures);
    ++erasures;

    deadOutPortBitVec.set(index);
  }

  // If there is nothing to remove, abort.
  if (deadInPortBitVec.none() && deadOutPortBitVec.none())
    return;

  // Erase arguments of the old module from liveSet to prevent from creating
  // dangling pointers.
  for (auto arg : module.getBodyBlock()->getArguments())
    liveElements.erase(arg);

  for (auto op : outputOp->getOperands())
    liveElements.erase(op);

  // Delete ports from the module.
  module.erasePorts(SmallVector<unsigned>(deadInPortBitVec.set_bits()),
                    SmallVector<unsigned>(deadOutPortBitVec.set_bits()));
  module.getBodyBlock()->eraseArguments(deadInPortBitVec);

  // Add arguments of the new module to liveSet.
  for (auto arg : module.getBodyBlock()->getArguments())
    liveElements.insert(arg);

  for (auto op : outputOp->getOperands())
    liveElements.insert(op);

  // Rewrite all instantiation of the module.
  for (auto *use : llvm::make_early_inc_range(instanceGraphNode->uses())) {
    auto instance = cast<InstanceOp>(*use->getInstance());

    ImplicitLocOpBuilder builder(instance.getLoc(), instance);
    // Replace old instance results with dummy constants.
    for (auto index : deadOutPortBitVec.set_bits())
      replaceInstanceResultWithConst(builder, index, instance);

    // Since we will rewrite instance op, it is necessary to remove old
    // instance results from liveSet.
    for (auto oldResult : instance.getResults())
      liveElements.erase(oldResult);

    auto newInstance = cloneWithErasePortsAndReplaceUses(
        instance, deadInPortBitVec, deadOutPortBitVec);

    for (auto newResult : newInstance.getResults())
      liveElements.insert(newResult);

    instanceGraph->replaceInstance(instance, newInstance);
    instance->erase();
  }

  numRemovedPorts += deadInPortBitVec.count() + deadOutPortBitVec.count();
}

void HWIMDeadCodeElim::rewriteModuleBody(HWModuleOp module) {

  // Walk the IR bottom-up when deleting operations.
  module.walk<mlir::WalkOrder::PostOrder, mlir::ReverseIterator>(
      [&](Operation *op) {
        // Connects to values that we found to be dead can be dropped.
        LLVM_DEBUG(llvm::dbgs() << "Visit: " << *op << "\n");

        // Remove non-sideeffect op using `isOpTriviallyDead`.
        // Skip instances - they're handled by
        // rewriteModuleSignature/eraseEmptyModule and also need erasure from
        // instanceGraph
        if (!isa<InstanceOp>(op) && mlir::isOpTriviallyDead(op) &&
            isAssumedDead(op)) {
          op->erase();
          ++numErasedOps;
        }
      });
}

void HWIMDeadCodeElim::eraseEmptyModule(HWModuleOp module) {
  // Non-executable modules are preserved as empty shells to avoid invalidating
  // symbol references that may point to them by name.
  if (!isBlockExecutable(module.getBodyBlock()))
    return;

  // If the module is not empty, just skip.
  if (!module.getBodyBlock()->without_terminator().empty())
    return;

  // It can also be the case that the only `hw.output` is nontrivial, also skip.
  if (module.getBodyBlock()->getTerminator()->getNumOperands() != 0)
    return;

  // We cannot delete public modules so generate a warning.
  if (module.isPublic()) {
    mlir::emitWarning(module.getLoc())
        << "module `" << module.getName()
        << "` is empty but cannot be removed because the module is public";
    return;
  }

  // Ok, the module is empty. Delete instances unless they have symbols.
  LLVM_DEBUG(llvm::dbgs() << "Erase " << module.getName() << "\n");
  igraph::InstanceGraphNode *instanceGraphNode =
      instanceGraph->lookup(module.getModuleNameAttr());

  SmallVector<Location> instancesWithSymbols;
  for (auto *use : llvm::make_early_inc_range(instanceGraphNode->uses())) {
    auto maybeInst = use->getInstance();
    if (!maybeInst)
      continue;

    auto instance = cast<InstanceOp>(maybeInst);
    if (instance.getInnerSym()) {
      instancesWithSymbols.push_back(instance.getLoc());
      continue;
    }
    use->erase();
    instance.erase();
  }

  // If there is an instance with a symbol, we don't delete the module itself.
  if (!instancesWithSymbols.empty()) {
    auto diag = module.emitWarning()
                << "module `" << module.getName()
                << "` is empty but cannot be removed because an instance is "
                   "referenced by name";
    diag.attachNote(FusedLoc::get(&getContext(), instancesWithSymbols))
        << "these are instances with symbols";
    return;
  }

  // We cannot delete alive modules.
  if (liveElements.contains(module))
    return;

  instanceGraph->erase(instanceGraphNode);
  ++numErasedModules;
}