//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/ProbeToSV.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HierPathCache.h"
#include "circt/Dialect/HW/InnerSymbolNamespace.h"
#include "circt/Dialect/HW/PortConverter.h"
#include "circt/Dialect/Probe/ProbeOps.h"
#include "circt/Dialect/Probe/ProbeTypes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Support/Namespace.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
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
bool containsProbeRef(Type type) {
  bool found = false;
  type.walk([&](Type nested) {
    if (isa<probe::RefType>(nested))
      found = true;
  });
  return found;
}

/// Add a field-zero inner symbol to `op`, or return its existing one.
StringAttr getOrAddInnerSym(hw::InnerSymbolOpInterface op,
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

/// Remove a Probe output from a module and its instances without replacing it
/// with any physical port.
class EraseProbeOutput : public hw::PortConversion {
public:
  using PortConversion::PortConversion;

  void mapInputSignals(OpBuilder &, Operation *, Value,
                       SmallVectorImpl<Value> &, ArrayRef<Backedge>) override {
    llvm_unreachable("Probe input ports must fail validation");
  }

  void mapOutputSignals(OpBuilder &, Operation *, Value instanceResult,
                        SmallVectorImpl<Value> &, ArrayRef<Backedge>) override {
    assert(instanceResult.use_empty() &&
           "Probe instance result must have no uses before port conversion");
  }

private:
  void buildInputSignals() override {
    llvm_unreachable("Probe input ports must fail validation");
  }
  void buildOutputSignals() override {}
};

class ProbePortConversionBuilder : public hw::PortConversionBuilder {
public:
  using PortConversionBuilder::PortConversionBuilder;

  FailureOr<std::unique_ptr<hw::PortConversion>>
  build(hw::PortInfo port) override {
    if (port.isOutput() && isa<probe::RefType>(port.type))
      return {std::make_unique<EraseProbeOutput>(converter, port)};
    return PortConversionBuilder::build(port);
  }
};

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
  LogicalResult validateProbeUse(Operation *op);
  LogicalResult rewrite();

  hw::InnerRefAttr getOrCreateSourceRef(
      probe::SendOp send,
      hw::InnerSymbolNamespaceCollection &innerSymbolNamespaces);
  hw::InnerRefAttr getOrCreateInstanceRef(
      hw::InstanceOp instance,
      hw::InnerSymbolNamespaceCollection &innerSymbolNamespaces);

  SymbolTableCollection symbolTables;
  SmallVector<ReadResolution> resolutions;
  SmallVector<hw::HWModuleOp> modulesWithProbeOutputs;
  SmallVector<probe::SendOp> sendOps;
  DenseMap<Operation *, hw::InnerRefAttr> sourceRefs;
};

LogicalResult LowerProbeToSVPass::validateModulePorts(hw::HWModuleLike module) {
  unsigned outputIndex = 0;
  bool hasProbeOutput = false;
  auto concreteModule = dyn_cast<hw::HWModuleOp>(*module);
  for (auto port : module.getPortList()) {
    if (!isa<probe::RefType>(port.type)) {
      if (containsProbeRef(port.type))
        return module.emitOpError(
            "the Probe dialect does not support nested Probe refs in module "
            "ports");
      if (port.isOutput())
        ++outputIndex;
      continue;
    }

    if (!concreteModule)
      return module.emitOpError(
          "Probe refs on external or generated module ports are not supported");
    if (!module.isPrivate())
      return module.emitOpError(
          "Probe output ports on public modules are not supported");

    auto output =
        cast<hw::OutputOp>(concreteModule.getBodyBlock()->getTerminator());
    auto send = output.getOperand(outputIndex).getDefiningOp<probe::SendOp>();
    if (!send)
      return output.emitOpError(
          "the Probe dialect requires a Probe output to be driven directly by "
          "probe.send; multi-level forwarding is not supported");
    hasProbeOutput = true;
    ++outputIndex;
  }
  if (hasProbeOutput)
    modulesWithProbeOutputs.push_back(cast<hw::HWModuleOp>(*module));
  return success();
}

FailureOr<ReadResolution> LowerProbeToSVPass::resolveRead(probe::ReadOp read) {
  Value input = read.getInput();

  if (auto send = input.getDefiningOp<probe::SendOp>())
    return ReadResolution{read, send, {}};

  auto result = cast<OpResult>(input);
  auto instance = cast<hw::InstanceOp>(result.getOwner());
  if (instance.getDoNotPrint())
    return read.emitOpError(
        "Probe-to-SV lowering cannot create an XMR through an hw.instance "
        "marked doNotPrint");

  auto *referenced = symbolTables.lookupNearestSymbolFrom(
      instance, instance.getModuleNameAttr());
  auto childModule = cast<hw::HWModuleOp>(referenced);

  auto output = cast<hw::OutputOp>(childModule.getBodyBlock()->getTerminator());
  auto send = cast<probe::SendOp>(
      output.getOperand(result.getResultNumber()).getDefiningOp());
  return ReadResolution{read, send, instance};
}

LogicalResult LowerProbeToSVPass::validateProbeUse(Operation *op) {
  bool hasProbeValue = llvm::any_of(op->getOperands(), [](Value value) {
    return containsProbeRef(value.getType());
  });
  hasProbeValue |= llvm::any_of(op->getResults(), [](Value value) {
    return containsProbeRef(value.getType());
  });
  if (!hasProbeValue ||
      isa<probe::SendOp, probe::ReadOp, hw::OutputOp, hw::InstanceOp>(op))
    return success();

  op->emitOpError(
      "the Probe dialect only permits Probe refs to flow through "
      "probe.send, probe.read, hw.output, and direct hw.instance results");
  return failure();
}

LogicalResult LowerProbeToSVPass::validate() {
  auto circuit = getOperation();

  for (auto module : circuit.getOps<hw::HWModuleLike>())
    if (failed(validateModulePorts(module)))
      return failure();

  SmallVector<probe::ReadOp> readOps;
  for (auto module : circuit.getOps<hw::HWModuleOp>()) {
    auto walkResult = module.walk([&](Operation *op) -> WalkResult {
      if (auto send = dyn_cast<probe::SendOp>(op))
        sendOps.push_back(send);
      else if (auto read = dyn_cast<probe::ReadOp>(op))
        readOps.push_back(read);

      if (failed(validateProbeUse(op)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      return failure();
  }

  for (auto send : sendOps)
    if (!hw::isHWValueType(send.getInput().getType()))
      return send.emitOpError(
                 "Probe-to-SV lowering requires an HW value payload, but got ")
             << send.getInput().getType();

  for (auto read : readOps) {
    auto resolution = resolveRead(read);
    if (failed(resolution))
      return failure();
    resolutions.push_back(*resolution);
  }
  return success();
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

LogicalResult LowerProbeToSVPass::rewrite() {
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

  // Remove Probe output ports and update all corresponding instances.
  auto &instanceGraph = getAnalysis<hw::InstanceGraph>();
  for (auto module : modulesWithProbeOutputs)
    if (failed(
            hw::PortConverter<ProbePortConversionBuilder>(instanceGraph, module)
                .run()))
      return failure();

  for (auto send : llvm::reverse(sendOps))
    send.erase();
  return success();
}

void LowerProbeToSVPass::runOnOperation() {
  if (failed(validate()))
    return signalPassFailure();
  if (failed(rewrite()))
    signalPassFailure();
}

} // namespace

std::unique_ptr<mlir::Pass> circt::createLowerProbeToSVPass() {
  return std::make_unique<LowerProbeToSVPass>();
}
