//===- HWRemoveUnusedPorts.cpp - Remove Dead Ports --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements HWRemoveUnusedPorts pass which removes dead ports by
// inspecting modules globally.
//
// 1. Input ports are dead if there is no user.
// 2. Output ports are dead if output values are constant, or results are not
//    used at any instance.
// 3. Normal operations are deleted as well since erased ports can introduce
//    extra dead code.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hw-remove-unused-ports"

using namespace llvm;
using namespace mlir;
using namespace circt;
using namespace hw;

/// Return true if the port is deletable. Specifically, we cannot delete ports
/// with symbols.
static bool isDeletablePort(PortInfo port) {
  return !port.sym || port.sym.getValue().empty();
}

/// Return true if the instance result is unused.
static bool isOutputPortUnused(InstanceRecord *record, unsigned index) {
  auto port = record->getInstance()->getResult(index);
  return port.use_empty();
}

/// Return true if the output port is deletable. We can delete the port if the
/// port doesn't have a symbol and its result value is not used at any instance.
static bool isDeadOutputPort(HWModuleOp module, InstanceGraphNode *node,
                             unsigned index) {
  if (!isDeletablePort(module.getOutputPort(index)))
    return false;

  return llvm::all_of(node->uses(), [&](InstanceRecord *record) {
    return isOutputPortUnused(record, index);
  });
}

/// Return true if the input port is deletable. We can delete the port if
/// there is no user and the port doesn't have a symbol.
static bool isDeletableInputPort(HWModuleOp module, unsigned index) {
  // If the port has use, we cannot delete the port.
  if (!module.getArgument(index).use_empty())
    return false;
  if (!isDeletablePort(module.getInOrInoutPort(index)))
    return false;

  return true;
}

namespace {
struct HWRemoveUnusedPortsPass
    : public sv::HWRemoveUnusedPortsBase<HWRemoveUnusedPortsPass> {
  void runOnOperation() override;

  // A helper function to visit all private modules in the post-order of
  // instance graph.
  void visitPrivateModules(bool doFinalize);
  void visitModule(HWModuleOp module, InstanceGraphNode *instanceGraphNode);

  /// This function actually rewrites module definitions and their instances.
  void finalize(HWModuleOp module, InstanceGraphNode *instanceGraphNode);

  void visitValue(Value value);
  void visitInputPort(StringAttr moduleName, unsigned index);
  void visitOutputPort(StringAttr moduleName, unsigned index);

  void addToWorklist(Value value) { worklist.insert(value); }

  // Return a place holder for the given value. Values created by this function
  // must be deleted at post-processing.
  Value getDummyValue(Value value);

  // Return a pair of module op and instance graph node if the module is a
  // private module. If not, return a pair of nullptr.
  std::pair<HWModuleOp, InstanceGraphNode *>
  getModuleIfPrivate(StringAttr moduleName);

  /// A worklist of values that might be dead. We have to use a set to avoid
  /// double free. SetVector is used to make the pass deterministic.
  llvm::SetVector<Value> worklist;

  InstanceGraph *instanceGraph;
  OpBuilder *builder;
};
} // namespace

// Return a place holder for the given value. Values created by this function
// must be deleted at post-processing.
Value HWRemoveUnusedPortsPass::getDummyValue(Value value) {
  builder->setInsertionPointAfterValue(value);
  return builder
      ->create<mlir::UnrealizedConversionCastOp>(
          value.getLoc(), TypeRange{value.getType()}, ValueRange{})
      .getResult(0);
}

// Return a pair of module op and instance graph node if the module is a
// private module. If not, return a pair of nullptr.
std::pair<HWModuleOp, InstanceGraphNode *>
HWRemoveUnusedPortsPass::getModuleIfPrivate(StringAttr moduleName) {
  auto *node = instanceGraph->lookup(moduleName);
  if (!node || !node->getModule() || node->getModule().isPublic())
    return {};

  auto module = dyn_cast<HWModuleOp>(node->getModule());
  return {module, node};
}

void HWRemoveUnusedPortsPass::visitOutputPort(StringAttr moduleName,
                                              unsigned index) {
  auto [module, node] = getModuleIfPrivate(moduleName);

  if (!module || !isDeadOutputPort(module, node, index))
    return;

  // If the output port is dead, replace its corresponding operand of output op.
  auto output = cast<OutputOp>(module.getBodyBlock()->getTerminator());
  auto operand = output->getOperand(index);
  output->setOperand(index, getDummyValue(operand));
  addToWorklist(operand);
}

void HWRemoveUnusedPortsPass::visitInputPort(StringAttr moduleName,
                                             unsigned index) {
  auto [module, node] = getModuleIfPrivate(moduleName);

  if (!module || !isDeletableInputPort(module, index))
    return;

  // If the input port is dead, traverse all uses and add their arguments
  // to the worklist.
  for (auto *use : node->uses()) {
    if (!use->getInstance())
      continue;
    auto instance = dyn_cast<InstanceOp>(*use->getInstance());

    if (!instance)
      continue;
    auto operand = instance.getOperand(index);
    instance->setOperand(index, getDummyValue(operand));
    addToWorklist(operand);
  }
}

void HWRemoveUnusedPortsPass::visitValue(Value value) {
  // If the value has an use, we cannot remove.
  if (!value.use_empty())
    return;

  // If the value is a result of instance, the result may be dead in every
  // instantiation.
  if (auto instance = value.getDefiningOp<HWInstanceLike>())
    return visitOutputPort(instance.referencedModuleNameAttr(),
                           value.cast<mlir::OpResult>().getResultNumber());

  if (auto inputPort = value.dyn_cast<BlockArgument>()) {
    auto hwmodule =
        dyn_cast<HWModuleLike>(inputPort.getParentBlock()->getParentOp());
    if (!hwmodule)
      return;
    return visitInputPort(hwmodule.moduleNameAttr(), inputPort.getArgNumber());
  }

  // Otherwise, delete the value if its defining op is dead.
  if (auto *op = value.getDefiningOp()) {
    // If the op is dead, add its operands to the worklist.
    if (isOpTriviallyDead(op)) {
      for (auto operand : op->getOperands())
        addToWorklist(operand);
      op->erase();
    }
  }
}

void HWRemoveUnusedPortsPass::visitPrivateModules(bool doFinalize) {
  for (auto *node : llvm::post_order(instanceGraph)) {
    if (!node || !node->getModule() || node->getModule().isPublic())
      continue;
    auto module = dyn_cast<HWModuleOp>(node->getModule());
    if (!module)
      continue;

    if (doFinalize)
      finalize(module, node);
    else
      visitModule(module, node);
  }
}

void HWRemoveUnusedPortsPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "===----- Remove unused ports -----===\n");
  instanceGraph = &getAnalysis<InstanceGraph>();
  OpBuilder theBuilder(&getContext());
  builder = &theBuilder;

  visitPrivateModules(/*doFinalize=*/false);

  while (!worklist.empty()) {
    auto value = worklist.pop_back_val();
    visitValue(value);
  }

  visitPrivateModules(/*doFinalize=*/true);
}

/// Remove elements at the specified indices from the input array, returning
/// the elements not mentioned.  The indices array is expected to be sorted
/// and unique.
template <typename T, typename RandomAccessRange>
static SmallVector<T>
removeElementsAtIndices(RandomAccessRange input,
                        ArrayRef<unsigned> indicesToDrop) {
  // Copy over the live chunks.
  size_t lastCopied = 0;
  SmallVector<T> result;
  result.reserve(input.size() - indicesToDrop.size());

  for (unsigned indexToDrop : indicesToDrop) {
    // If we skipped over some valid elements, copy them over.
    if (indexToDrop > lastCopied) {
      result.append(input.begin() + lastCopied, input.begin() + indexToDrop);
      lastCopied = indexToDrop;
    }
    // Ignore this value so we don't copy it in the next iteration.
    ++lastCopied;
  }

  // If there are live elements at the end, copy them over.
  if (lastCopied < input.size())
    result.append(input.begin() + lastCopied, input.end());

  return result;
}

void HWRemoveUnusedPortsPass::finalize(HWModuleOp module,
                                       InstanceGraphNode *instanceGraphNode) {
  LLVM_DEBUG(llvm::dbgs() << "Prune ports of module: " << module.getName()
                          << "\n");
  SmallVector<unsigned> removalInputPortIndexes;
  SmallVector<unsigned> removalOutputPortIndexes;

  SmallDenseSet<Operation *, 4> deadOperations;
  auto addMayDeadValue = [&](Value v) {
    if (auto *op = v.getDefiningOp())
      deadOperations.insert(op);
  };

  for (auto index : llvm::seq(0u, module.getNumResults()))
    if (isDeadOutputPort(module, instanceGraphNode, index))
      removalOutputPortIndexes.push_back(index);

  auto output = cast<OutputOp>(module.getBodyBlock()->getTerminator());
  builder->setInsertionPoint(output);
  if (!removalOutputPortIndexes.empty()) {
    auto newOutput = removeElementsAtIndices<Value>(output.operands(),
                                                    removalOutputPortIndexes);
    builder->create<hw::OutputOp>(output.getLoc(), newOutput);
    for (auto index : removalOutputPortIndexes)
      addMayDeadValue(output.getOperand(index));
    output.erase();
  }

  // Traverse input ports.
  for (auto index : llvm::seq(0u, module.getNumArguments()))
    if (isDeletableInputPort(module, index))
      removalInputPortIndexes.push_back(index);

  // If there is nothing to remove, abort.
  if (removalInputPortIndexes.empty() && removalOutputPortIndexes.empty())
    return;

  // Delete ports from the module.
  module.erasePorts(removalInputPortIndexes, removalOutputPortIndexes);

  // Delete arguments. It is necessary to remove the argument in the reverse
  // order of `removalInputPortIndexes`.
  for (auto arg : llvm::reverse(removalInputPortIndexes))
    module.getBody().eraseArgument(arg);

  // Rewrite all uses.
  for (auto *use : instanceGraphNode->uses()) {
    auto instance = dyn_cast<InstanceOp>(*use->getInstance());
    if (!instance)
      continue;
    for (auto c : removalInputPortIndexes)
      addMayDeadValue(instance.getOperand(c));

    builder->setInsertionPoint(instance);
    // Create a new instance op without unused ports.
    auto newInstance = instance.erasePorts(*builder, removalInputPortIndexes,
                                           removalOutputPortIndexes);

    instanceGraph->replaceInstance(instance, newInstance);
    // Remove old one.
    instance.erase();
  }

  for (auto *op : deadOperations)
    if (isOpTriviallyDead(op))
      op->erase();

  numRemovedPorts += removalInputPortIndexes.size();
  numRemovedPorts += removalOutputPortIndexes.size();
}

void HWRemoveUnusedPortsPass::visitModule(
    HWModuleOp module, InstanceGraphNode *instanceGraphNode) {
  LLVM_DEBUG(llvm::dbgs() << "Preprocess module: " << module.getName() << "\n");
  // These track port indexes that can be erased.
  SmallVector<unsigned> removalInputPortIndexes;
  SmallVector<unsigned> removalOutputPortIndexes;

  // This tracks constant values of output ports.
  SmallVector<llvm::Optional<APInt>> outputPortConstants;
  auto ports = module.getPorts();

  // Traverse output ports.
  auto output = cast<OutputOp>(module.getBodyBlock()->getTerminator());
  for (auto e : llvm::enumerate(ports.outputs)) {
    unsigned index = e.index();
    auto port = e.value();
    if (!isDeletablePort(port))
      continue;

    // If the output port has no user at any instance, the port is dead.
    if (llvm::all_of(instanceGraphNode->uses(), [&](auto record) {
          return isOutputPortUnused(record, index);
        })) {
      outputPortConstants.push_back(None);
      removalOutputPortIndexes.push_back(index);
      auto result = output.getOperand(index);
      // Replace a curresponding operand with a dummy value.
      output.setOperand(index, getDummyValue(result));
      addToWorklist(result);
      continue;
    }

    // If the output value is constant, we can forward it into caller side.
    auto *src = output.getOperand(index).getDefiningOp();
    if (!isa_and_nonnull<hw::ConstantOp, sv::ConstantXOp>(src))
      continue;

    if (auto constant = dyn_cast<hw::ConstantOp>(src))
      outputPortConstants.push_back(constant.value());
    else
      outputPortConstants.push_back(None);

    removalOutputPortIndexes.push_back(index);
    addToWorklist(output->getOperand(index));
  }

  for (auto index : llvm::seq(0u, module.getNumArguments()))
    if (isDeletableInputPort(module, index))
      removalInputPortIndexes.push_back(index);

  // If there is nothing to remove, abort.
  if (removalInputPortIndexes.empty() && removalOutputPortIndexes.empty())
    return;

  // Rewrite all uses.
  for (auto *use : instanceGraphNode->uses()) {
    auto instance = dyn_cast<InstanceOp>(*use->getInstance());
    if (!instance)
      continue;

    for (auto [index, constant] :
         llvm::zip(removalOutputPortIndexes, outputPortConstants)) {
      auto result = instance.getResult(index);
      if (result.use_empty())
        continue;

      builder->setInsertionPoint(instance);
      Value value;
      if (constant)
        value = builder->create<hw::ConstantOp>(result.getLoc(), *constant);
      else
        value =
            builder->create<sv::ConstantXOp>(result.getLoc(), result.getType());

      result.replaceAllUsesWith(value);
    }

    // Replace instance arguments with dummy values.
    for (auto inputPort : removalInputPortIndexes) {
      auto operand = instance.getOperand(inputPort);
      instance.setOperand(inputPort, getDummyValue(operand));
      addToWorklist(operand);
    }
  }
}

std::unique_ptr<mlir::Pass> circt::sv::createHWRemoveUnusedPortsPass() {
  return std::make_unique<HWRemoveUnusedPortsPass>();
}
