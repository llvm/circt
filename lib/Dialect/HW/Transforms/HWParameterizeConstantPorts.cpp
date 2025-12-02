//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass parametizes constant ports on private modules if every instance
// has a constant as ports (e.g. hart id). This makes it possible to identify
// constant ports without inter-module analysis.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/HW/PortImplementation.h"
#include "circt/Support/InstanceGraph.h"
#include "circt/Support/Namespace.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hw-parameterize-constant-ports"

namespace circt {
namespace hw {
#define GEN_PASS_DEF_HWPARAMETERIZECONSTANTPORTS
#include "circt/Dialect/HW/Passes.h.inc"
} // namespace hw
} // namespace circt

using namespace circt;
using namespace hw;

/// Helper to extract the attribute value and defining operation from a
/// constant or param.value op.
static std::pair<Attribute, Operation *>
getAttributeAndDefiningOp(InstanceOp inst, unsigned portIndex) {
  Value value = inst.getInputs()[portIndex];
  auto *op = value.getDefiningOp();
  if (!op)
    return {};
  if (auto constOp = dyn_cast<hw::ConstantOp>(op))
    return {constOp.getValueAttr(), op};
  if (auto paramOp = dyn_cast<hw::ParamValueOp>(op))
    return {paramOp.getValueAttr(), op};
  return {};
}

/// Check if all instances have constant values for a given port.
static bool allInstancesHaveConstantForPort(igraph::InstanceGraphNode *node,
                                            unsigned portIndex) {
  if (node->noUses())
    return false;

  for (auto *instRecord : node->uses()) {
    auto inst = dyn_cast<InstanceOp>(instRecord->getInstance().getOperation());
    if (!inst || !getAttributeAndDefiningOp(inst, portIndex).first)
      return false;
  }
  return true;
}

namespace {
struct HWParameterizeConstantPortsPass
    : public circt::hw::impl::HWParameterizeConstantPortsBase<
          HWParameterizeConstantPortsPass> {
  void runOnOperation() override;

private:
  void processModule(HWModuleOp module, igraph::InstanceGraphNode *node,
                     hw::InstanceGraph &instanceGraph);
};
} // namespace

void HWParameterizeConstantPortsPass::processModule(
    HWModuleOp module, igraph::InstanceGraphNode *node,
    hw::InstanceGraph &instanceGraph) {
  // Only process private modules with instances
  if (!module.isPrivate() || node->noUses())
    return;

  // Find input ports that are constant across all instances
  hw::ModulePortInfo portInfo(module.getPortList());
  SmallVector<hw::PortInfo> inputPorts(portInfo.getInputs());
  SmallVector<unsigned> portsToParameterize;

  for (auto [idx, port] : llvm::enumerate(inputPorts)) {
    // Skip non-input ports and ports with symbols (could be forced)
    if (port.dir != ModulePort::Direction::Input || port.getSym())
      continue;

    if (allInstancesHaveConstantForPort(node, idx))
      portsToParameterize.push_back(idx);
  }

  if (portsToParameterize.empty())
    return;

  LLVM_DEBUG(llvm::dbgs() << "Parameterizing " << portsToParameterize.size()
                          << " ports in module " << module.getModuleName()
                          << "\n");

  OpBuilder builder(module.getContext());
  builder.setInsertionPointToStart(module.getBodyBlock());

  // Create parameters and replace port uses
  SmallVector<Attribute> newParameters;
  Namespace paramNamespace;
  if (auto existingParams = module.getParameters()) {
    newParameters.append(existingParams.begin(), existingParams.end());
    for (auto param : existingParams)
      paramNamespace.newName(cast<ParamDeclAttr>(param).getName().str());
  }

  // Map from port index to parameter name
  DenseMap<unsigned, StringAttr> portToParamName;

  for (unsigned portIdx : portsToParameterize) {
    auto port = inputPorts[portIdx];

    // Create a parameter name based on the port name
    auto paramNameAttr =
        builder.getStringAttr(paramNamespace.newName(port.name.str()));
    portToParamName[portIdx] = paramNameAttr;

    // Create parameter declaration without default value
    auto paramDecl = ParamDeclAttr::get(paramNameAttr, port.type);
    newParameters.push_back(paramDecl);

    // Replace uses of the port argument with a param.value operation
    auto paramRef = ParamDeclRefAttr::get(paramNameAttr, port.type);
    auto paramValueOp =
        ParamValueOp::create(builder, module.getLoc(), port.type, paramRef);

    // Replace all uses of the port argument with the param.value operation
    module.getBodyBlock()->getArgument(portIdx).replaceAllUsesWith(
        paramValueOp);
  }

  // Update module parameters
  module.setParametersAttr(builder.getArrayAttr(newParameters));

  // Remove the parameterized ports from the module signature
  module.modifyPorts({}, {}, portsToParameterize, {});

  // Erase block arguments in reverse order to maintain indices
  for (auto idx : llvm::reverse(portsToParameterize))
    module.getBodyBlock()->eraseArgument(idx);

  // Build new port names for instances (excluding removed ports)
  DenseSet<unsigned> portsToRemoveSet(portsToParameterize.begin(),
                                      portsToParameterize.end());
  SmallVector<Attribute> newPortNames;
  for (auto [idx, port] : llvm::enumerate(portInfo.getInputs()))
    if (!portsToRemoveSet.count(idx))
      newPortNames.push_back(port.name);

  ArrayAttr newPortNamesAttr = builder.getArrayAttr(newPortNames);

  // Update all instances of this module
  for (auto *instRecord : node->uses()) {
    auto inst = dyn_cast<InstanceOp>(instRecord->getInstance().getOperation());
    // Skip non-InstanceOp users (e.g., InstanceChoiceOp or other instance-like
    // operations).
    if (!inst)
      continue;

    builder.setInsertionPoint(inst);

    // Collect existing parameters and add new ones for removed ports
    SmallVector<Attribute> instParams;
    if (auto existingParams = inst.getParameters())
      instParams.append(existingParams.begin(), existingParams.end());

    for (unsigned portIdx : portsToParameterize) {
      auto [paramValueAttr, constOp] = getAttributeAndDefiningOp(inst, portIdx);
      assert(paramValueAttr && "expected constant or param value");
      auto paramDecl =
          ParamDeclAttr::get(builder.getContext(), portToParamName[portIdx],
                             inputPorts[portIdx].type, paramValueAttr);
      instParams.push_back(paramDecl);

      // Delete the constant or param.value op if it's only used by this
      // instance.
      if (constOp->hasOneUse()) {
        constOp->dropAllUses();
        constOp->erase();
      }
    }

    // Build new input list excluding parameterized ports
    SmallVector<Value> newInputs;
    for (auto [idx, input] : llvm::enumerate(inst.getInputs()))
      if (!portsToRemoveSet.count(idx))
        newInputs.push_back(input);

    // Create new instance with updated parameters and inputs
    auto newInst = InstanceOp::create(
        builder, inst.getLoc(), inst.getResultTypes(),
        inst.getInstanceNameAttr(), inst.getModuleNameAttr(), newInputs,
        newPortNamesAttr, inst.getResultNamesAttr(),
        builder.getArrayAttr(instParams), inst.getInnerSymAttr(),
        inst.getDoNotPrintAttr());

    // Replace old instance
    instanceGraph.replaceInstance(inst, newInst);
    inst.replaceAllUsesWith(newInst.getResults());
    inst.erase();
  }
}

void HWParameterizeConstantPortsPass::runOnOperation() {
  auto &instanceGraph = getAnalysis<hw::InstanceGraph>();

  // Process all HW modules in inverse post-order (top-down).
  instanceGraph.walkInversePostOrder([&](igraph::InstanceGraphNode &node) {
    if (auto module =
            dyn_cast_or_null<HWModuleOp>(node.getModule().getOperation()))
      processModule(module, &node, instanceGraph);
  });

  // The instance graph is updated during the pass and remains valid.
  markAnalysesPreserved<hw::InstanceGraph>();
}
