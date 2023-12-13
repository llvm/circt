//===- FIRRTLInstanceImplementation.cpp - Utilities for instance-like ops -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLInstanceImplementation.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"

using namespace circt;
using namespace circt::firrtl;

LogicalResult
instance_like_impl::verifyReferencedModule(Operation *instanceOp,
                                           SymbolTableCollection &symbolTable,
                                           mlir::FlatSymbolRefAttr moduleName) {
  auto module = instanceOp->getParentOfType<FModuleOp>();
  auto referencedModule =
      symbolTable.lookupNearestSymbolFrom<FModuleLike>(instanceOp, moduleName);
  if (!referencedModule) {
    return instanceOp->emitOpError("invalid symbol reference");
  }

  // Check this is not a class.
  if (isa<ClassOp /* ClassLike */>(referencedModule))
    return instanceOp->emitOpError("must instantiate a module not a class")
               .attachNote(referencedModule.getLoc())
           << "class declared here";

  // Check that this instance doesn't recursively instantiate its wrapping
  // module.
  if (referencedModule == module) {
    auto diag = instanceOp->emitOpError()
                << "is a recursive instantiation of its containing module";
    return diag.attachNote(module.getLoc())
           << "containing module declared here";
  }

  // Small helper add a note to the original declaration.
  auto emitNote = [&](InFlightDiagnostic &&diag) -> InFlightDiagnostic && {
    diag.attachNote(referencedModule->getLoc())
        << "original module declared here";
    return std::move(diag);
  };

  // Check that all the attribute arrays are the right length up front.  This
  // lets us safely use the port name in error messages below.
  size_t numResults = instanceOp->getNumResults();
  size_t numExpected = referencedModule.getNumPorts();
  if (numResults != numExpected) {
    return emitNote(instanceOp->emitOpError()
                    << "has a wrong number of results; expected " << numExpected
                    << " but got " << numResults);
  }
  auto portDirections =
      instanceOp->getAttrOfType<IntegerAttr>("portDirections");
  if (portDirections.getValue().getBitWidth() != numExpected)
    return emitNote(
        instanceOp->emitOpError("the number of port directions should be "
                                "equal to the number of results"));

  auto portNames = instanceOp->getAttrOfType<ArrayAttr>("portNames");
  if (portNames.size() != numExpected)
    return emitNote(
        instanceOp->emitOpError("the number of port names should be "
                                "equal to the number of results"));

  auto portAnnotations =
      instanceOp->getAttrOfType<ArrayAttr>("portAnnotations");
  if (portAnnotations.size() != numExpected)
    return emitNote(
        instanceOp->emitOpError("the number of result annotations should be "
                                "equal to the number of results"));

  // Check that the port names match the referenced module.
  if (portNames != referencedModule.getPortNamesAttr()) {
    // We know there is an error, try to figure out whats wrong.
    auto moduleNames = referencedModule.getPortNamesAttr();
    // First compare the sizes:
    if (portNames.size() != moduleNames.size()) {
      return emitNote(instanceOp->emitOpError()
                      << "has a wrong number of directions; expected "
                      << moduleNames.size() << " but got " << portNames.size());
    }
    // Next check the values:
    for (size_t i = 0; i != numResults; ++i) {
      if (portNames[i] != moduleNames[i]) {
        return emitNote(instanceOp->emitOpError()
                        << "name for port " << i << " must be "
                        << moduleNames[i] << ", but got " << portNames[i]);
      }
    }
    llvm_unreachable("should have found something wrong");
  }

  // Check that the types match.
  for (size_t i = 0; i != numResults; i++) {
    auto resultType = instanceOp->getResult(i).getType();
    auto expectedType = referencedModule.getPortType(i);
    if (resultType != expectedType) {
      return emitNote(instanceOp->emitOpError()
                      << "result type for " << portNames[i] << " must be "
                      << expectedType << ", but got " << resultType);
    }
  }

  // Check that the port directions are consistent with the referenced module's.
  if (portDirections != referencedModule.getPortDirectionsAttr()) {
    // We know there is an error, try to figure out whats wrong.
    auto moduleDirectionAttr = referencedModule.getPortDirectionsAttr();
    // First compare the sizes:
    auto expectedWidth = moduleDirectionAttr.getValue().getBitWidth();
    auto actualWidth = portDirections.getValue().getBitWidth();
    if (expectedWidth != actualWidth) {
      return emitNote(instanceOp->emitOpError()
                      << "has a wrong number of directions; expected "
                      << expectedWidth << " but got " << actualWidth);
    }
    // Next check the values.
    auto instanceDirs = direction::unpackAttribute(portDirections);
    auto moduleDirs = direction::unpackAttribute(moduleDirectionAttr);
    for (size_t i = 0; i != numResults; ++i) {
      if (instanceDirs[i] != moduleDirs[i]) {
        return emitNote(instanceOp->emitOpError()
                        << "direction for " << portNames[i] << " must be \""
                        << direction::toString(moduleDirs[i])
                        << "\", but got \""
                        << direction::toString(instanceDirs[i]) << "\"");
      }
    }
    llvm_unreachable("should have found something wrong");
  }

  return success();
}
