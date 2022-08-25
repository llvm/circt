//===- Feedthrough.cpp - Add feedthrough ports to specified modules -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace circt::hw;

namespace {
struct FeedthroughPass : public circt::FeedthroughBase<FeedthroughPass> {
  void runOnOperation() override;
};
} // namespace

void FeedthroughPass::runOnOperation() {
  // Some basic preconditions.
  if (sourceModule.empty()) {
    emitError(getOperation().getLoc()) << "source module is required.";
    return signalPassFailure();
  }

  if (destModule.empty()) {
    emitError(getOperation().getLoc()) << "destination module is required.";
    return signalPassFailure();
  }

  // Set up the builder, instance graph, and starting point.
  OpBuilder builder(getOperation());

  InstanceGraph &instanceGraph = getAnalysis<InstanceGraph>();

  InstanceGraphNode *sourceNode =
      instanceGraph.lookup(StringAttr::get(&getContext(), sourceModule));

  if (sourceNode->getModule().isPublic()) {
    emitError(getOperation().getLoc())
        << "module is public and cannot be modified";
    return signalPassFailure();
  }

  // Collect PortInfos to add as inputs and outputs throughout the modules.
  SmallVector<PortInfo> portInfo =
      getAllModulePortInfos(sourceNode->getModule());

  SmallVector<PortInfo> inputsToAdd;
  SmallVector<PortInfo> outputsToAdd;
  for (PortInfo port : portInfo) {
    for (const std::string &sourcePort : sourcePorts) {
      if (port.getName().equals(sourcePort)) {
        inputsToAdd.push_back(PortInfo{
            StringAttr::get(&getContext(), "feedthrough_" + port.getName()),
            PortDirection::INPUT, port.type});
        outputsToAdd.push_back(PortInfo{
            StringAttr::get(&getContext(), "feedthrough_" + port.getName()),
            PortDirection::OUTPUT, port.type});
      }
    }
  }

  // Punch the outputs out of the source module.
  HWMutableModuleLike sourceModule =
      cast<HWMutableModuleLike>(sourceNode->getModule().getOperation());

  SmallVector<std::pair<StringAttr, Value>> outputValuesToAdd;
  Block &bodyBlock = sourceModule->getRegion(0).front();
  auto sourceArgNames =
      sourceModule.getArgNames().getAsValueRange<StringAttr>();
  for (auto &[i, port] : llvm::enumerate(
           llvm::make_range(sourceArgNames.begin(), sourceArgNames.end())))
    for (const std::string &sourcePort : sourcePorts)
      if (port.equals(sourcePort))
        outputValuesToAdd.emplace_back(
            StringAttr::get(&getContext(), "feedthrough_" + port),
            bodyBlock.getArgument(i));

  sourceModule.appendOutputs(outputValuesToAdd);

  // Punch the outputs out of the source module's instance(s).
  SmallVector<Value> feedthroughValues;
  for (auto *inst : sourceNode->uses()) {
    HWInstanceLike instance = inst->getInstance();

    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointAfter(instance.getOperation());

    SmallVector<Value> operands(instance->getOperands());
    auto newInstance = builder.create<InstanceOp>(
        instance.getLoc(), inst->getTarget()->getModule(),
        instance.instanceNameAttr(), operands);

    // Note that this iteration works because we append the new feedthroughs on
    // the end of the module/instance lists.
    for (size_t i = 0, e = instance->getNumResults(); i < e; ++i)
      instance->getResult(i).replaceAllUsesWith(newInstance->getResult(i));

    // Grab the new outputs since we'll need to send them as inputs to the next
    // instance we replace.
    for (size_t i = instance->getNumResults(), e = newInstance->getNumResults();
         i < e; ++i)
      feedthroughValues.push_back(newInstance->getResult(i));

    instanceGraph.replaceInstance(instance, newInstance);
    instance.erase();
  }

  // Now move down the list of intermediates, punching inputs and outputs.
  for (const std::string &intermediate : innerModules) {
    // Lookup the module.
    auto *intermediateModule =
        instanceGraph.lookup(StringAttr::get(&getContext(), intermediate));

    auto mutableModule = cast<HWMutableModuleLike>(
        intermediateModule->getModule().getOperation());

    // Create the new input ports, add block arguments, and remember the values.
    auto numInputs = mutableModule.getNumInputs();
    SmallVector<std::pair<unsigned, PortInfo>> inputPortsToAdd;
    Block &block = mutableModule->getRegion(0).front();
    outputValuesToAdd.clear();
    for (auto input : inputsToAdd) {
      inputPortsToAdd.emplace_back(numInputs, input);
      auto newArg = block.addArgument(input.type, mutableModule.getLoc());
      outputValuesToAdd.emplace_back(input.name, newArg);
    }
    mutableModule.insertPorts(inputPortsToAdd, {});

    // Connect the new inputs to the new outputs. This also mutates the ports.
    mutableModule.appendOutputs(outputValuesToAdd);

    // Rewrite the instances.
    for (auto *inst : intermediateModule->uses()) {
      HWInstanceLike instance = inst->getInstance();

      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointAfter(instance.getOperation());

      SmallVector<Value> operands(instance->getOperands());
      for (auto feedthrough : feedthroughValues)
        operands.push_back(feedthrough);
      auto newInstance = builder.create<InstanceOp>(
          instance.getLoc(), inst->getTarget()->getModule(),
          instance.instanceNameAttr(), operands);

      // Note that this iteration works because we append the new feedthroughs
      // on the end of the module/instance lists.
      for (size_t i = 0, e = instance->getNumResults(); i < e; ++i)
        instance->getResult(i).replaceAllUsesWith(newInstance->getResult(i));

      // Grab the new outputs since we'll need to send them as inputs to the
      // next instance we replace.
      feedthroughValues.clear();
      for (size_t i = instance->getNumResults(),
                  e = newInstance->getNumResults();
           i < e; ++i)
        feedthroughValues.push_back(newInstance->getResult(i));

      instanceGraph.replaceInstance(instance, newInstance);
      instance.erase();
    }
  }
}

std::unique_ptr<Pass> circt::createFeedthroughPass() {
  return std::make_unique<FeedthroughPass>();
}
