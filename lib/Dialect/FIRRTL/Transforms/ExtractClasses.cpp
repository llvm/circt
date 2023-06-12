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
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
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
  Value getOrCreateObjectValue(OpResult instanceOutput, InstanceOp instance,
                               OpBuilder &builder, IRMapping &mapping,
                               SmallVectorImpl<Operation *> &opsToErase);
  ObjectOp getOrCreateObject(InstanceOp instance, OpBuilder &builder,
                             IRMapping &mapping,
                             SmallVectorImpl<Operation *> &opsToErase);
  void extractClass(FModuleOp moduleOp);
  void updateInstances(FModuleOp moduleOp);

  InstanceGraph *instanceGraph;
  DenseMap<Operation *, llvm::BitVector> portsToErase;
  DenseMap<OpResult, Value> cachedObjectValues;
  DenseMap<InstanceOp, ObjectOp> cachedObjects;
};
} // namespace

/// Helper class to capture details about a property.
struct Property {
  size_t index;
  StringRef name;
  Type type;
  Location loc;
};

/// Helper to get or create a Value from an ObjectFieldOp. This consults a cache
/// to avoid re-creating ObjectFieldOps. An ObjectOp is looked up or created,
/// and the field corresponding to the instance output is accessed. Note that
/// this is able to work locally with just the instance and the OpResult, and
/// the rest of the pass ensures the correct ClassOp is created.
Value ExtractClassesPass::getOrCreateObjectValue(
    OpResult instanceOutput, InstanceOp instance, OpBuilder &builder,
    IRMapping &mapping, SmallVectorImpl<Operation *> &opsToErase) {
  // Check if this ObjectField has already been created, and return it if so.
  auto cachedObjectValue = cachedObjectValues.find(instanceOutput);
  if (cachedObjectValue != cachedObjectValues.end())
    return cachedObjectValue->getSecond();

  // Get the result number for the InstanceOp output.
  unsigned resultNum = instanceOutput.getResultNumber();

  // Get the field type.
  Type fieldType = instance.getResult(resultNum).getType();

  // Get the ObjectOp to extract a field from.
  ObjectOp object = getOrCreateObject(instance, builder, mapping, opsToErase);

  // Get the field path.
  StringAttr resultName = instance.getPortName(resultNum);
  ArrayAttr fieldPath =
      builder.getArrayAttr(FlatSymbolRefAttr::get(resultName));

  // Construct the ObjectFieldOp.
  auto fieldValue = builder.create<ObjectFieldOp>(instance.getLoc(), fieldType,
                                                  object, fieldPath);

  // Cache it for potential future lookups.
  cachedObjectValues[instanceOutput] = fieldValue;

  return fieldValue;
}

/// Helper to get or create an ObjectOp. This consults a cache to avoid
/// re-creating ObjectOps. The actual parameters are computed by finding the
/// assignments to the InstanceOp's inputs. The property ops involved are
/// cloned, mapped, and marked for erasure as appropriate. The ObjectOp is then
/// created with the appropriate type, class name, and the actual parameters.
/// Note that this is able to work locally with just the instance, and the rest
/// of the pass ensures the correct ClassOp is created.
ObjectOp ExtractClassesPass::getOrCreateObject(
    InstanceOp instance, OpBuilder &builder, IRMapping &mapping,
    SmallVectorImpl<Operation *> &opsToErase) {
  // Check if this ObjectField has already been created, and return it if so.
  auto cachedObject = cachedObjects.find(instance);
  if (cachedObject != cachedObjects.end())
    return cachedObject->getSecond();

  // Build up the ObjectOp's actual parameters.
  SmallVector<Value> actualParams;
  for (size_t i = 0; i < instance.getNumResults(); ++i) {
    // Skip outputs and non-Property inputs.
    if (instance.getPortDirection(i) == Direction::Out)
      continue;
    auto result = dyn_cast<FIRRTLPropertyValue>(instance.getResult(i));
    if (!result)
      continue;

    // Get the assignment to the input property.
    PropAssignOp propassign = getPropertyAssignment(result);

    // The source value will be mapped into the actual parameter.
    Value inputValue = propassign.getSrc();

    // The property assignment must be deleted.
    opsToErase.push_back(propassign);

    // If the property was assigned from a property op, copy that over to the
    // ClassOp as well. This may need to walk property ops, but for now only
    // constant ops exist.
    if (auto *definingOp = inputValue.getDefiningOp())
      builder.clone(*definingOp, mapping);

    // Lookup the mapping for the input value, and use this as an actual
    // parameter.
    actualParams.push_back(mapping.lookup(inputValue));
  }

  // Get the Object type.
  auto objectType =
      ClassType::get(instance.getContext(), instance.getModuleNameAttr());

  // Get the Object's class name.
  auto objectClass = instance.getModuleNameAttr().getAttr();

  // Construct the ObjectOp.
  auto object = builder.create<ObjectOp>(instance.getLoc(), objectType,
                                         objectClass, actualParams);

  // Cache it for potential future lookups.
  cachedObjects[instance] = object;

  return object;
}

/// Potentiall extract an OM class from a FIRRTL module which may contain
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

    // Mark the property assign to be erased.
    opsToErase.push_back(getPropertyAssignment(outputValue));

    // If the Value is defined by an Operation, copy it into the ClassOp.
    if (auto *op = originalValue.getDefiningOp()) {
      // InstanceOps are handled specially, by creating ObjectOps and extracting
      // an ObjectFieldOp. This will take care of re-using ObjectOps and
      // ObjectFieldOps, updating the mapping, and keeping track of related ops
      // that can be erased. The original value is mapped to the ObjectFieldOp.
      if (auto instance = dyn_cast<InstanceOp>(op)) {
        Value fieldValue =
            getOrCreateObjectValue(cast<OpResult>(originalValue), instance,
                                   builder, mapping, opsToErase);
        mapping.map(originalValue, fieldValue);
      } else {
        // For all other ops, copy the defining op into the body, and
        // map from the old Value to the new Value. This may need to walk
        // property ops in order to copy them into the ClassOp, but for now only
        // constant ops exist. Mark the property op to be erased.
        builder.clone(*op, mapping);

        // Property defining ops should be erased after being copied over.
        opsToErase.push_back(op);
      }
    }

    // Create the ClassFieldOp using the mapping to find the appropriate Value.
    Value fieldValue = mapping.lookup(originalValue);
    builder.create<ClassFieldOp>(originalValue.getLoc(), outputProperty.name,
                                 fieldValue);
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
  llvm::BitVector modulePortsToErase = portsToErase[moduleOp];
  InstanceGraphNode *instanceGraphNode = instanceGraph->lookup(moduleOp);

  // If there are no ports to erase, nothing to do.
  if (!modulePortsToErase.empty() && !modulePortsToErase.any())
    return;

  // Clean up instances of the FModuleOp.
  for (InstanceRecord *node : instanceGraphNode->uses()) {
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
      for (Operation *user : oldInstance.getResult(propertyIndex).getUsers()) {
        assert(isa<FConnectLike>(user) &&
               "expected property pins to be used in property assignments");
        user->erase();
      }
    }

    // Erase the original instance.
    oldInstance.erase();
    node->erase();
  }

  // If all ports are properties, remove the FModuleOp completely.
  if (!modulePortsToErase.empty() && modulePortsToErase.all()) {
    moduleOp.erase();
    instanceGraph->erase(instanceGraphNode);
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
