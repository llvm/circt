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
      instanceOp->getAttrOfType<mlir::DenseBoolArrayAttr>("portDirections");
  if (static_cast<size_t>(portDirections.size()) != numExpected)
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

  auto domainInfo = instanceOp->getAttrOfType<ArrayAttr>("domainInfo");
  if (domainInfo.size() != numExpected)
    return emitNote(
        instanceOp->emitOpError()
        << "has a wrong number of port domain info attributes; expected "
        << numExpected << ", got " << domainInfo.size());

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
    auto expectedWidth = moduleDirectionAttr.size();
    auto actualWidth = portDirections.size();
    if (expectedWidth != actualWidth) {
      return emitNote(instanceOp->emitOpError()
                      << "has a wrong number of directions; expected "
                      << expectedWidth << " but got " << actualWidth);
    }
    // Next check the values.
    auto instanceDirs = portDirections;
    for (size_t i = 0; i != numResults; ++i) {
      if (instanceDirs[i] != moduleDirectionAttr[i]) {
        return emitNote(instanceOp->emitOpError()
                        << "direction for " << portNames[i] << " must be \""
                        << direction::toString(moduleDirectionAttr[i])
                        << "\", but got \""
                        << direction::toString(instanceDirs[i]) << "\"");
      }
    }
    llvm_unreachable("should have found something wrong");
  }

  // Check the domain info matches.
  for (size_t i = 0; i < numResults; ++i) {
    auto portDomainInfo = domainInfo[i];
    auto modulePortDomainInfo = referencedModule.getDomainInfoAttrForPort(i);
    if (portDomainInfo != modulePortDomainInfo)
      return emitNote(instanceOp->emitOpError()
                      << "domain info for " << portNames[i] << " must be "
                      << modulePortDomainInfo << ", but got "
                      << portDomainInfo);
  }

  // Check that the instance op lists the correct layer requirements.
  auto instanceLayers = instanceOp->getAttrOfType<ArrayAttr>("layers");
  auto moduleLayers = referencedModule.getLayersAttr();
  if (instanceLayers != moduleLayers)
    return emitNote(instanceOp->emitOpError()
                    << "layers must be " << moduleLayers << ", but got "
                    << instanceLayers);

  return success();
}
