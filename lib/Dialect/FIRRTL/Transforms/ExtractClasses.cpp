//===- ExtractClasses.cpp - Extract OM classes ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ExtractClasses pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/OM/OMOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace circt;
using namespace circt::firrtl;
using namespace circt::om;

namespace {
struct ExtractClassesPass : public ExtractClassesBase<ExtractClassesPass> {
  void runOnOperation() override;

private:
  void extractClass(FModuleOp moduleOp);
  void updateInstances(FModuleOp moduleOp);

  InstanceGraph *instanceGraph;
  DenseMap<Operation *, llvm::BitVector> portsToErase;
};
} // namespace

/// Helper class to capture details about a property.
struct Property {
  size_t index;
  StringRef name;
  Type type;
  Location loc;
};

/// Potentially extract an OM class from a FIRRTL module which may contain
/// properties.
void ExtractClassesPass::extractClass(FModuleOp moduleOp) {
  // Map from Values in the FModuleOp to Values in the ClassOp.
  IRMapping mapping;

  // Remember ports and operations to clean up when done.
  portsToErase[moduleOp] = llvm::BitVector(moduleOp.getNumPorts());
  SmallVector<Operation *> opsToErase;

  // Collect information about input and output properties. Mark property ports
  // to be erased.
  SmallVector<Property> inputProperties;
  SmallVector<Property> outputProperties;
  for (auto [index, port] : llvm::enumerate(moduleOp.getPorts())) {
    if (!isa<PropertyType>(port.type))
      continue;

    portsToErase[moduleOp].set(index);

    if (port.isInput())
      inputProperties.push_back({index, port.name, port.type, port.loc});

    if (port.isOutput())
      outputProperties.push_back({index, port.name, port.type, port.loc});
  }

  // If the FModuleOp has no properties, nothing to do.
  if (inputProperties.empty() && outputProperties.empty())
    return;

  OpBuilder builder = OpBuilder::atBlockEnd(getOperation().getBody(0));

  // Collect the parameter names from input properties.
  SmallVector<StringRef> formalParamNames;
  for (auto inputProperty : inputProperties)
    formalParamNames.push_back(inputProperty.name);

  // Construct the ClassOp with the FModuleOp name and parameter names.
  auto classOp = builder.create<ClassOp>(moduleOp.getLoc(), moduleOp.getName(),
                                         formalParamNames);

  // Construct the ClassOp body with block arguments for each input property,
  // updating the mapping to map from the input property to the block argument.
  Block *classBody = &classOp.getRegion().emplaceBlock();
  for (auto inputProperty : inputProperties) {
    BlockArgument parameterValue =
        classBody->addArgument(inputProperty.type, inputProperty.loc);
    BlockArgument inputValue = moduleOp.getArgument(inputProperty.index);
    mapping.map(inputValue, parameterValue);
  }

  // Construct ClassFieldOps for each output property.
  builder.setInsertionPointToStart(classBody);
  for (auto outputProperty : outputProperties) {
    // Get the Value driven to the property to use for this ClassFieldOp.
    auto outputValue =
        cast<FIRRTLPropertyValue>(moduleOp.getArgument(outputProperty.index));
    Value originalValue = getDriverFromConnect(outputValue);

    // If the Value is defined by an Operation that hasn't been copied yet, copy
    // that into the body, and map from the old Value to the new Value. This may
    // need to walk property ops in order to copy them into the ClassOp, but for
    // now only constant ops exist. Mark the property op to be erased.
    if (!mapping.contains(originalValue)) {
      if (auto *op = originalValue.getDefiningOp()) {
        builder.clone(*op, mapping);

        // InstanceOps are handled specially for now, but any other property
        // defining ops should be erased after being copied over.
        if (!isa<InstanceOp>(op))
          opsToErase.push_back(op);
      }
    }

    // Create the ClassFieldOp using the mapping to find the appropriate Value.
    Value fieldValue = mapping.lookup(originalValue);
    builder.create<ClassFieldOp>(originalValue.getLoc(), outputProperty.name,
                                 fieldValue);

    // Eagerly erase the property assign, since it is done now.
    getPropertyAssignment(outputValue).erase();
  }

  // Clean up the FModuleOp by removing property ports and operations. This
  // first erases opsToErase in the order they were added, so property
  // assignments are erased before value defining ops. Then it erases ports.
  for (auto *op : opsToErase)
    op->erase();
  moduleOp.erasePorts(portsToErase[moduleOp]);
}

/// Clean up InstanceOps of any FModuleOps with properties.
void ExtractClassesPass::updateInstances(FModuleOp moduleOp) {
  OpBuilder builder(&getContext());
  const llvm::BitVector &modulePortsToErase = portsToErase[moduleOp];
  InstanceGraphNode *instanceGraphNode = instanceGraph->lookup(moduleOp);

  // If there are no ports to erase, nothing to do.
  if (!modulePortsToErase.empty() && !modulePortsToErase.any())
    return;

  // Clean up instances of the FModuleOp.
  for (InstanceRecord *node :
       llvm::make_early_inc_range(instanceGraphNode->uses())) {
    // Get the original InstanceOp.
    InstanceOp oldInstance = cast<InstanceOp>(node->getInstance());
    builder.setInsertionPointAfter(oldInstance);

    // If some but not all ports are properties, create a new instance without
    // the property pins.
    if (!modulePortsToErase.all()) {
      InstanceOp newInstance =
          oldInstance.erasePorts(builder, portsToErase[moduleOp]);
      instanceGraph->replaceInstance(oldInstance, newInstance);
    }

    // Clean up uses of property pins. This amounts to erasing property
    // assignments for now.
    for (int propertyIndex : modulePortsToErase.set_bits()) {
      for (Operation *user : llvm::make_early_inc_range(
               oldInstance.getResult(propertyIndex).getUsers())) {
        assert(isa<FConnectLike>(user) &&
               "expected property pins to be used in property assignments");
        user->erase();
      }
    }

    // Erase the original instance.
    node->erase();
    oldInstance.erase();
  }

  // If all ports are properties, remove the FModuleOp completely.
  if (!modulePortsToErase.empty() && modulePortsToErase.all()) {
    instanceGraph->erase(instanceGraphNode);
    moduleOp.erase();
  }
}

/// Extract OM classes from FIRRTL modules with properties.
void ExtractClassesPass::runOnOperation() {
  // Get the CircuitOp.
  auto circuits = getOperation().getOps<CircuitOp>();
  if (circuits.empty())
    return;
  CircuitOp circuit = *circuits.begin();

  // Get the FIRRTL instance graph.
  instanceGraph = &getAnalysis<InstanceGraph>();

  // Walk all FModuleOps to potentially extract an OM class if the FModuleOp
  // contains properties.
  for (auto moduleOp : llvm::make_early_inc_range(circuit.getOps<FModuleOp>()))
    extractClass(moduleOp);

  // Clean up InstanceOps of any FModuleOps with properties. This is done after
  // the classes are extracted to avoid extra bookeeping as InstanceOps are
  // cleaned up.
  for (auto moduleOp : llvm::make_early_inc_range(circuit.getOps<FModuleOp>()))
    updateInstances(moduleOp);

  // Mark analyses preserved, since we keep the instance graph up to date.
  markAllAnalysesPreserved();

  // Reset pass state.
  instanceGraph = nullptr;
}

std::unique_ptr<mlir::Pass> circt::firrtl::createExtractClassesPass() {
  return std::make_unique<ExtractClassesPass>();
}
