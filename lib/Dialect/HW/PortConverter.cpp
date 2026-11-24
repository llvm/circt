//===- PortConverter.cpp - Module I/O rewriting utility ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/PortConverter.h"

#include <numeric>

using namespace circt;
using namespace hw;

/// Return a attribute with the specified suffix appended.
static StringAttr append(StringAttr base, const Twine &suffix) {
  if (suffix.isTriviallyEmpty())
    return base;
  auto *context = base.getContext();
  return StringAttr::get(context, base.getValue() + suffix);
}

/// Return the per-direction index at which a new port of direction `dir`,
/// created for `origPort` of `mod`, is inserted.
/// This is the number of original ports of direction `dir` before `origPort`,
/// so the new ports of the module keep the order of their original ports.
static unsigned getInsertionIndex(HWMutableModuleLike mod, PortInfo origPort,
                                  ModulePort::Direction dir) {
  if (origPort.dir == dir)
    return origPort.argNum;
  auto type = mod.getHWModuleType();
  size_t portId = origPort.isOutput()
                      ? type.getPortIdForOutputId(origPort.argNum)
                      : type.getPortIdForInputId(origPort.argNum);
  // The ports before `origPort` not of its direction are of direction `dir`.
  return portId - origPort.argNum;
}

namespace {

/// We consider non-caught ports to be ad-hoc signaling or 'untouched'. (Which
/// counts as a signaling protocol if one squints pretty hard). We mostly do
/// this since it allows us a more consistent internal API.
class UntouchedPortConversion : public PortConversion {
public:
  UntouchedPortConversion(PortConverterImpl &converter, hw::PortInfo origPort)
      : PortConversion(converter, origPort) {
    // Set the "RTTI flag" to true (see comment in header for this variable).
    isUntouchedFlag = true;
  }

  void mapInputSignals(OpBuilder &b, Operation *inst, Value instValue,
                       SmallVectorImpl<Value> &newOperands,
                       ArrayRef<Backedge> newResults) override {
    newOperands[portInfo.argNum] = instValue;
  }
  void mapOutputSignals(OpBuilder &b, Operation *inst, Value instValue,
                        SmallVectorImpl<Value> &newOperands,
                        ArrayRef<Backedge> newResults) override {
    instValue.replaceAllUsesWith(newResults[portInfo.argNum]);
  }

private:
  void buildInputSignals() override {
    Value newValue = converter.createNewInput(
        origPort, "", origPort.type, portInfo, PortAttrPolicy::Preserve);
    if (body)
      body->getArgument(origPort.argNum).replaceAllUsesWith(newValue);
  }

  void buildOutputSignals() override {
    Value output;
    if (body)
      output = body->getTerminator()->getOperand(origPort.argNum);
    converter.createNewOutput(origPort, "", origPort.type, output, portInfo,
                              PortAttrPolicy::Preserve);
  }

  hw::PortInfo portInfo;
};

} // namespace

FailureOr<std::unique_ptr<PortConversion>>
PortConversionBuilder::build(hw::PortInfo port) {
  // Default builder is the 'untouched' port conversion which will simply
  // pass ports through unmodified.
  return {std::make_unique<UntouchedPortConversion>(converter, port)};
}

PortConverterImpl::PortConverterImpl(igraph::InstanceGraphNode *moduleNode)
    : moduleNode(moduleNode), b(moduleNode->getModule()->getContext()) {
  mod = dyn_cast<hw::HWMutableModuleLike>(*moduleNode->getModule());
  assert(mod && "PortConverter only works on HWMutableModuleLike");

  if (mod->getNumRegions() == 1 && mod->getRegion(0).hasOneBlock()) {
    body = &mod->getRegion(0).front();
    terminator = body->getTerminator();
  }
}

Value PortConverterImpl::createNewInput(PortInfo origPort, const Twine &suffix,
                                        Type type, PortInfo &newPort,
                                        PortAttrPolicy attrPolicy) {
  DictionaryAttr attrs = attrPolicy == PortAttrPolicy::Preserve
                             ? origPort.attrs
                             : DictionaryAttr();
  newPort = PortInfo{
      {append(origPort.name, suffix), type, ModulePort::Direction::Input},
      newInputs.size(),
      attrs,
      origPort.loc};
  newInputs.emplace_back(
      getInsertionIndex(mod, origPort, ModulePort::Direction::Input), newPort);

  if (!body)
    return {};
  return body->addArgument(type, origPort.loc);
}

void PortConverterImpl::createNewOutput(PortInfo origPort, const Twine &suffix,
                                        Type type, Value output,
                                        PortInfo &newPort,
                                        PortAttrPolicy attrPolicy) {
  DictionaryAttr attrs = attrPolicy == PortAttrPolicy::Preserve
                             ? origPort.attrs
                             : DictionaryAttr();
  newPort = PortInfo{
      {append(origPort.name, suffix), type, ModulePort::Direction::Output},
      newOutputs.size(),
      attrs,
      origPort.loc};
  newOutputs.emplace_back(
      getInsertionIndex(mod, origPort, ModulePort::Direction::Output), newPort);

  if (!body)
    return;

  OpBuilder::InsertionGuard g(b);
  b.setInsertionPointToStart(body);
  terminator->insertOperands(terminator->getNumOperands(), output);
}

LogicalResult PortConverterImpl::run() {
  ModulePortInfo ports(mod.getPortList());

  bool foundLoweredPorts = false;

  auto createPortLowering = [&](PortInfo port) {
    auto &loweredPorts = port.dir == ModulePort::Direction::Output
                             ? loweredOutputs
                             : loweredInputs;

    auto loweredPort = ssb->build(port);
    if (failed(loweredPort))
      return failure();

    foundLoweredPorts |= !(*loweredPort)->isUntouched();
    loweredPorts.emplace_back(std::move(*loweredPort));

    if (failed(loweredPorts.back()->init()))
      return failure();

    return success();
  };

  // Dispatch the port conversion builder on the I/O of the module.
  for (PortInfo port : ports)
    if (failed(createPortLowering(port)))
      return failure();

  // Bail early if we didn't find anything to convert.
  if (!foundLoweredPorts) {
    // Memory optimization.
    loweredInputs.clear();
    loweredOutputs.clear();
    return success();
  }

  // Lower the ports -- this mutates the body directly and builds the port
  // lists.  Lowering in port order keeps the lists in ascending index order.
  for (PortInfo port : ports)
    (port.isOutput() ? loweredOutputs : loweredInputs)[port.argNum]
        ->lowerPort();
  [[maybe_unused]] auto byIndex = [](const auto &a, const auto &b) {
    return a.first < b.first;
  };
  assert(llvm::is_sorted(newInputs, byIndex) &&
         llvm::is_sorted(newOutputs, byIndex) &&
         "new ports must be created in the order of their original ports");

  // Set up vectors to erase _all_ the ports. Each new port is inserted at the
  // position of its original port, so the rebuilt port list keeps the order
  // of the original one.
  SmallVector<unsigned> inputsToErase(mod.getNumInputPorts());
  std::iota(inputsToErase.begin(), inputsToErase.end(), 0);
  SmallVector<unsigned> outputsToErase(mod.getNumOutputPorts());
  std::iota(outputsToErase.begin(), outputsToErase.end(), 0);

  mod.modifyPorts(newInputs, newOutputs, inputsToErase, outputsToErase);

  if (body) {
    // We should only erase the original arguments. New ones were appended
    // with the `createInput` method call.
    body->eraseArguments([&ports](BlockArgument arg) {
      return arg.getArgNumber() < ports.sizeInputs();
    });

    // And erase the first ports.sizeOutputs operands from the terminator.
    terminator->eraseOperands(0, ports.sizeOutputs());
  }

  // Rewrite instances pointing to this module.
  for (auto *instance : moduleNode->uses()) {
    auto instanceLike = instance->getInstance<hw::HWInstanceLike>();
    if (!instanceLike)
      continue;
    hw::InstanceOp hwInstance = dyn_cast_or_null<hw::InstanceOp>(*instanceLike);
    if (!hwInstance) {
      return instanceLike->emitOpError(
          "This code only converts hw.instance instances - ask your friendly "
          "neighborhood compiler engineers to implement support for something "
          "like an hw::HWMutableInstanceLike interface");
    }
    updateInstance(hwInstance);
  }

  // Memory optimization -- we don't need these anymore.
  newInputs.clear();
  newOutputs.clear();
  return success();
}

void PortConverterImpl::updateInstance(hw::InstanceOp inst) {
  ImplicitLocOpBuilder b(inst.getLoc(), inst);
  BackedgeBuilder beb(b, inst.getLoc());
  ModulePortInfo ports(mod.getPortList());

  // Create backedges for the future instance results so the signal mappers can
  // use the future results as values.
  SmallVector<Backedge> newResults;
  for (PortInfo outputPort : ports.getOutputs())
    newResults.push_back(beb.get(outputPort.type));

  // Map the operands.
  SmallVector<Value> newOperands(ports.sizeInputs(), {});
  for (size_t oldOpIdx = 0, e = inst.getNumOperands(); oldOpIdx < e; ++oldOpIdx)
    loweredInputs[oldOpIdx]->mapInputSignals(
        b, inst, inst->getOperand(oldOpIdx), newOperands, newResults);

  // Map the results.
  for (size_t oldResIdx = 0, e = inst.getNumResults(); oldResIdx < e;
       ++oldResIdx)
    loweredOutputs[oldResIdx]->mapOutputSignals(
        b, inst, inst->getResult(oldResIdx), newOperands, newResults);

  // Clone the instance. We cannot just modifiy the existing one since the
  // result types might have changed types and number of them.
  assert(llvm::none_of(newOperands, [](Value v) { return !v; }));
  b.setInsertionPointAfter(inst);
  auto newInst =
      InstanceOp::create(b, mod, inst.getInstanceNameAttr(), newOperands,
                         inst.getParameters(), inst.getInnerSymAttr());
  newInst->setDialectAttrs(inst->getDialectAttrs());
  if (auto doNotPrint = inst.getDoNotPrintAttr())
    newInst.setDoNotPrintAttr(doNotPrint);

  // Assign the backedges to the new results.
  for (auto [idx, be] : llvm::enumerate(newResults))
    be.setValue(newInst.getResult(idx));

  // Erase the old instance.
  inst.erase();
}
