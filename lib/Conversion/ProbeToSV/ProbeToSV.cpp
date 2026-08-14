//===- ProbeToSV.cpp - Probe to SV lowering ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/ProbeToSV.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HierPathCache.h"
#include "circt/Dialect/HW/InnerSymbolNamespace.h"
#include "circt/Dialect/Probe/ProbeOps.h"
#include "circt/Dialect/Probe/ProbeTypes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Support/Namespace.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"

#define DEBUG_TYPE "lower-probe-to-sv"

namespace circt {
#define GEN_PASS_DEF_LOWERPROBETOSV
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace circt;
using namespace mlir;

namespace {

struct ReadResolution {
  probe::ReadOp read;
  probe::SendOp send;
  hw::InstanceOp instance;
};

/// Return true if `type` is, or recursively contains, a Probe ref.
static bool containsProbeRef(Type type) {
  bool found = false;
  type.walk([&](Type nested) {
    if (isa<probe::RefType>(nested))
      found = true;
  });
  return found;
}

/// Add a field-zero inner symbol to `op`, or return its existing one.
static StringAttr
getOrAddInnerSym(hw::InnerSymbolOpInterface op,
                 hw::InnerSymbolNamespaceCollection &namespaces,
                 hw::HWModuleOp module, StringRef nameHint) {
  auto oldAttr = op.getInnerSymAttr();
  if (oldAttr)
    if (auto name = oldAttr.getSymName())
      return name;

  auto &ns = namespaces.get(module);
  auto name = StringAttr::get(op.getContext(), ns.newName(nameHint));
  SmallVector<hw::InnerSymPropertiesAttr> properties;
  properties.push_back(hw::InnerSymPropertiesAttr::get(name));
  if (oldAttr)
    llvm::append_range(properties, oldAttr.getProps());
  op.setInnerSymbolAttr(hw::InnerSymAttr::get(op.getContext(), properties));
  return name;
}

/// Clone an instance while dropping selected results. All operation attributes
/// are retained, including non-structural/custom attributes.
static hw::InstanceOp cloneWithoutResults(hw::InstanceOp instance,
                                          const llvm::BitVector &eraseResults) {
  SmallVector<Type> resultTypes;
  SmallVector<Attribute> resultNames;
  for (auto [index, result] : llvm::enumerate(instance.getResults())) {
    if (eraseResults.test(index))
      continue;
    resultTypes.push_back(result.getType());
    resultNames.push_back(instance.getResultNames()[index]);
  }

  OpBuilder builder(instance);
  OperationState state(instance.getLoc(), hw::InstanceOp::getOperationName());
  state.addOperands(instance.getOperands());
  state.addTypes(resultTypes);
  state.addAttributes(instance->getAttrs());
  auto newInstance = cast<hw::InstanceOp>(builder.create(state));
  newInstance.setResultNamesAttr(builder.getArrayAttr(resultNames));

  unsigned newIndex = 0;
  for (auto [oldIndex, oldResult] : llvm::enumerate(instance.getResults())) {
    if (eraseResults.test(oldIndex))
      continue;
    oldResult.replaceAllUsesWith(newInstance.getResult(newIndex++));
  }
  return newInstance;
}

class LowerProbeToSVPass
    : public circt::impl::LowerProbeToSVBase<LowerProbeToSVPass> {
public:
  LowerProbeToSVPass() = default;
  LowerProbeToSVPass(const LowerProbeToSVPass &pass) : Base(pass) {}

  void runOnOperation() override;

private:
  LogicalResult validate();
  FailureOr<ReadResolution> resolveRead(probe::ReadOp read);
  LogicalResult validateModulePorts(hw::HWModuleLike module);
  LogicalResult validateProbeUses();
  void rewrite();

  hw::InnerRefAttr getOrCreateSourceRef(
      probe::SendOp send,
      hw::InnerSymbolNamespaceCollection &innerSymbolNamespaces);
  hw::InnerRefAttr getOrCreateInstanceRef(
      hw::InstanceOp instance,
      hw::InnerSymbolNamespaceCollection &innerSymbolNamespaces);

  SymbolTableCollection symbolTables;
  SmallVector<ReadResolution> resolutions;
  DenseMap<hw::HWModuleOp, SmallVector<unsigned>> probeOutputs;
  SmallVector<probe::SendOp> sendOps;
  SmallVector<hw::InstanceOp> instancesWithProbeResults;
  DenseMap<Operation *, hw::InnerRefAttr> sourceRefs;
};

LogicalResult LowerProbeToSVPass::validateModulePorts(hw::HWModuleLike module) {
  unsigned outputIndex = 0;
  for (auto port : module.getPortList()) {
    auto refType = dyn_cast<probe::RefType>(port.type);
    if (!refType) {
      if (containsProbeRef(port.type))
        return module.emitOpError(
            "nested Probe refs in module ports are not supported by "
            "Probe-to-SV lowering");
      if (port.isOutput())
        ++outputIndex;
      continue;
    }

    if (!port.isOutput())
      return module.emitOpError(
          "Probe-to-SV lowering only supports direct Probe output ports");
    if (!hw::isHWValueType(refType.getElementType()))
      return module.emitOpError(
                 "Probe-to-SV lowering does not support payload type ")
             << refType.getElementType()
             << "; lower it to an HW value type before this pass";

    auto concreteModule = dyn_cast<hw::HWModuleOp>(*module);
    if (!concreteModule)
      return module.emitOpError(
          "Probe output ports on external or generated modules are not "
          "supported by Probe-to-SV lowering");
    if (!module.isPrivate())
      return module.emitOpError(
          "public Probe output ports require an external Probe ABI and "
          "cannot be lowered to an SV XMR");

    auto output =
        cast<hw::OutputOp>(concreteModule.getBodyBlock()->getTerminator());
    auto send = output.getOperand(outputIndex).getDefiningOp<probe::SendOp>();
    if (!send)
      return output.emitOpError(
          "Probe output must be driven directly by probe.send; forwarding "
          "Probe refs across multiple module levels is not supported");
    probeOutputs[concreteModule].push_back(outputIndex);
    ++outputIndex;
  }
  return success();
}

FailureOr<ReadResolution> LowerProbeToSVPass::resolveRead(probe::ReadOp read) {
  auto parentModule = read->getParentOfType<hw::HWModuleOp>();
  Value input = read.getInput();

  if (auto send = input.getDefiningOp<probe::SendOp>()) {
    if (send->getParentOfType<hw::HWModuleOp>() != parentModule)
      return read.emitOpError(
          "local Probe source must be in the same hw.module as the read");
    return ReadResolution{read, send, {}};
  }

  auto result = dyn_cast<OpResult>(input);
  auto instance =
      result ? dyn_cast<hw::InstanceOp>(result.getOwner()) : hw::InstanceOp{};
  if (!instance)
    return read.emitOpError(
        "Probe input must come from a local probe.send or a direct child "
        "hw.instance output");
  if (instance.getDoNotPrint())
    return read.emitOpError(
        "Probe input cannot originate from an hw.instance marked doNotPrint");

  auto *referenced = symbolTables.lookupNearestSymbolFrom(
      instance, instance.getModuleNameAttr());
  auto childModule = dyn_cast_or_null<hw::HWModuleOp>(referenced);
  if (!childModule)
    return read.emitOpError(
        "Probe input must originate in a concrete child hw.module");
  if (result.getResultNumber() >= childModule.getNumOutputPorts())
    return read.emitOpError("Probe instance result has no matching output");

  auto output = cast<hw::OutputOp>(childModule.getBodyBlock()->getTerminator());
  auto send = output.getOperand(result.getResultNumber())
                  .getDefiningOp<probe::SendOp>();
  if (!send)
    return read.emitOpError(
        "Probe instance result is not driven directly by probe.send in the "
        "child module; multi-level Probe forwarding is not supported");
  return ReadResolution{read, send, instance};
}

LogicalResult LowerProbeToSVPass::validateProbeUses() {
  auto circuit = getOperation();
  auto walkResult = circuit.walk([&](Operation *op) -> WalkResult {
    bool hasProbeValue = llvm::any_of(op->getOperands(), [](Value value) {
      return containsProbeRef(value.getType());
    });
    hasProbeValue |= llvm::any_of(op->getResults(), [](Value value) {
      return containsProbeRef(value.getType());
    });
    if (!hasProbeValue)
      return WalkResult::advance();

    if (isa<probe::SendOp, probe::ReadOp, hw::OutputOp, hw::InstanceOp>(op))
      return WalkResult::advance();

    op->emitOpError(
        "Probe refs may only flow through probe.send, probe.read, hw.output, "
        "and direct hw.instance results during Probe-to-SV lowering");
    return WalkResult::interrupt();
  });
  if (walkResult.wasInterrupted())
    return failure();

  for (auto send : sendOps) {
    if (!hw::isHWValueType(send.getInput().getType()))
      return send.emitOpError(
                 "Probe-to-SV lowering does not support payload type ")
             << send.getInput().getType()
             << "; lower it to an HW value type before this pass";
    for (Operation *user : send.getRef().getUsers())
      if (!isa<probe::ReadOp, hw::OutputOp>(user))
        return user->emitOpError(
            "unsupported use of a Probe ref produced by probe.send");
  }

  for (auto instance : instancesWithProbeResults) {
    for (auto result : instance.getResults()) {
      if (!isa<probe::RefType>(result.getType()))
        continue;
      for (Operation *user : result.getUsers()) {
        if (isa<probe::ReadOp>(user))
          continue;
        if (isa<hw::OutputOp>(user))
          return user->emitOpError(
              "forwarding a Probe ref through multiple module levels is not "
              "supported by Probe-to-SV lowering");
        return user->emitOpError(
            "Probe instance results may only be consumed by probe.read");
      }
    }
  }
  return success();
}

LogicalResult LowerProbeToSVPass::validate() {
  auto circuit = getOperation();

  for (auto module : circuit.getOps<hw::HWModuleLike>())
    if (failed(validateModulePorts(module)))
      return failure();

  circuit.walk([&](probe::SendOp send) { sendOps.push_back(send); });
  circuit.walk([&](hw::InstanceOp instance) {
    if (llvm::any_of(instance.getResultTypes(),
                     [](Type type) { return containsProbeRef(type); }))
      instancesWithProbeResults.push_back(instance);
  });

  if (failed(validateProbeUses()))
    return failure();

  LogicalResult result = success();
  circuit.walk([&](probe::ReadOp read) {
    if (failed(result))
      return;
    auto resolution = resolveRead(read);
    if (failed(resolution)) {
      result = failure();
      return;
    }
    if (!hw::isHWValueType(read.getResult().getType())) {
      read.emitOpError("Probe-to-SV lowering does not support payload type ")
          << read.getResult().getType()
          << "; lower it to an HW value type before this pass";
      result = failure();
      return;
    }
    resolutions.push_back(*resolution);
  });
  return result;
}

hw::InnerRefAttr LowerProbeToSVPass::getOrCreateSourceRef(
    probe::SendOp send,
    hw::InnerSymbolNamespaceCollection &innerSymbolNamespaces) {
  auto [it, inserted] = sourceRefs.try_emplace(send.getOperation());
  if (!inserted)
    return it->second;

  auto module = send->getParentOfType<hw::HWModuleOp>();
  auto &ns = innerSymbolNamespaces.get(module);
  auto name = StringAttr::get(&getContext(), ns.newName("probe"));
  ImplicitLocOpBuilder builder(send.getLoc(), send);
  hw::WireOp::create(builder, send.getInput(), name,
                     hw::InnerSymAttr::get(name));
  it->second = hw::InnerRefAttr::get(module.getModuleNameAttr(), name);
  return it->second;
}

hw::InnerRefAttr LowerProbeToSVPass::getOrCreateInstanceRef(
    hw::InstanceOp instance,
    hw::InnerSymbolNamespaceCollection &innerSymbolNamespaces) {
  auto module = instance->getParentOfType<hw::HWModuleOp>();
  auto innerSym = cast<hw::InnerSymbolOpInterface>(instance.getOperation());
  auto name = getOrAddInnerSym(innerSym, innerSymbolNamespaces, module,
                               instance.getInstanceName());
  return hw::InnerRefAttr::get(module.getModuleNameAttr(), name);
}

void LowerProbeToSVPass::rewrite() {
  auto circuit = getOperation();
  Namespace circuitNamespace;
  circuitNamespace.add(circuit);
  hw::HierPathCache pathCache(
      &circuitNamespace,
      OpBuilder::InsertPoint(circuit.getBody(), circuit.getBody()->begin()));
  hw::InnerSymbolNamespaceCollection innerSymbolNamespaces;

  for (auto resolution : resolutions) {
    SmallVector<Attribute> path;
    if (resolution.instance)
      path.push_back(
          getOrCreateInstanceRef(resolution.instance, innerSymbolNamespaces));
    path.push_back(
        getOrCreateSourceRef(resolution.send, innerSymbolNamespaces));

    ImplicitLocOpBuilder builder(resolution.read.getLoc(), resolution.read);
    auto hierPath = pathCache.getOrCreatePath(builder.getArrayAttr(path),
                                              resolution.read.getLoc());
    auto ref = FlatSymbolRefAttr::get(hierPath.getSymNameAttr());
    auto xmr = sv::XMRRefOp::create(
        builder, hw::InOutType::get(resolution.read.getResult().getType()), ref,
        builder.getStringAttr(""));
    auto value = sv::ReadInOutOp::create(builder, xmr);
    resolution.read.replaceAllUsesWith(value.getResult());
    resolution.read.erase();
  }

  // Remove Probe results from instances after all reads have been rewritten.
  for (auto instance : instancesWithProbeResults) {
    llvm::BitVector eraseResults(instance.getNumResults());
    for (auto [index, result] : llvm::enumerate(instance.getResults()))
      if (isa<probe::RefType>(result.getType()))
        eraseResults.set(index);
    cloneWithoutResults(instance, eraseResults);
    instance.erase();
  }

  // Remove Probe output operands and ports from their defining modules.
  for (auto &[module, outputs] : probeOutputs) {
    auto output = cast<hw::OutputOp>(module.getBodyBlock()->getTerminator());
    for (unsigned index : llvm::reverse(outputs))
      output->eraseOperand(index);
    module.erasePorts({}, outputs);
  }

  for (auto send : llvm::reverse(sendOps))
    send.erase();
}

void LowerProbeToSVPass::runOnOperation() {
  if (failed(validate()))
    return signalPassFailure();
  rewrite();
}

} // namespace

std::unique_ptr<mlir::Pass> circt::createLowerProbeToSVPass() {
  return std::make_unique<LowerProbeToSVPass>();
}
