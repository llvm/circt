//===- FIRRTLOpInterfaces.cpp - Implement the FIRRTL op interfaces --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement the FIRRTL operation interfaces.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;
using namespace llvm;
using namespace circt::firrtl;

LogicalResult circt::firrtl::verifyModuleLikeOpInterface(FModuleLike module) {
  // Verify port types first.  This is used as the basis for the number of
  // ports required everywhere else.
  auto portTypes = module.getPortTypesAttr();
  if (!portTypes || llvm::any_of(portTypes.getValue(), [](Attribute attr) {
        return !attr.isa<TypeAttr>();
      }))
    return module.emitOpError("requires valid port types");

  auto numPorts = portTypes.size();

  // Verify the port dirctions.
  auto portDirections = module.getPortDirectionsAttr();
  if (!portDirections)
    return module.emitOpError("requires valid port direction");
  // TODO: bitwidth is 1 when there are no ports, since APInt previously did not
  // support 0 bit widths.
  auto bitWidth = portDirections.getValue().getBitWidth();
  if (bitWidth != numPorts)
    return module.emitOpError("requires ") << numPorts << " port directions";

  // Verify the port names.
  auto portNames = module.getPortNamesAttr();
  if (!portNames)
    return module.emitOpError("requires valid port names");
  if (portNames.size() != numPorts)
    return module.emitOpError("requires ") << numPorts << " port names";
  if (llvm::any_of(portNames.getValue(),
                   [](Attribute attr) { return !attr.isa<StringAttr>(); }))
    return module.emitOpError("port names should all be string attributes");

  // Verify the port annotations.
  auto portAnnotations = module.getPortAnnotationsAttr();
  if (!portAnnotations)
    return module.emitOpError("requires valid port annotations");
  // TODO: it seems weird to allow empty port annotations.
  if (!portAnnotations.empty() && portAnnotations.size() != numPorts)
    return module.emitOpError("requires ") << numPorts << " port annotations";
  // TODO: Move this into an annotation verifier.
  for (auto annos : portAnnotations.getValue()) {
    auto arrayAttr = annos.dyn_cast<ArrayAttr>();
    if (!arrayAttr)
      return module.emitOpError(
          "requires port annotations be array attributes");
    if (llvm::any_of(arrayAttr.getValue(), [](Attribute attr) {
          return !attr.isa<DictionaryAttr>();
        }))
      return module.emitOpError(
          "annotations must be dictionaries or subannotations");
  }

  // Verify the port symbols.
  auto portSymbols = module.getPortSymbolsAttr();
  if (!portSymbols)
    return module.emitOpError("requires valid port symbols");
  if (!portSymbols.empty() && portSymbols.size() != numPorts)
    return module.emitOpError("requires ") << numPorts << " port symbols";
  if (llvm::any_of(portSymbols.getValue(), [](Attribute attr) {
        return !attr || !attr.isa<hw::InnerSymAttr>();
      }))
    return module.emitOpError("port symbols should all be InnerSym attributes");

  // Verify the port locations.
  auto portLocs = module.getPortLocationsAttr();
  if (!portLocs)
    return module.emitOpError("requires valid port locations");
  if (portLocs.size() != numPorts)
    return module.emitOpError("requires ") << numPorts << " port locations";
  if (llvm::any_of(portLocs.getValue(), [](Attribute attr) {
        return !attr || !attr.isa<LocationAttr>();
      }))
    return module.emitOpError("port symbols should all be location attributes");

  // Verify the body.
  if (module->getNumRegions() != 1)
    return module.emitOpError("requires one region");

  return success();
}

#include "circt/Dialect/FIRRTL/FIRRTLOpInterfaces.cpp.inc"
