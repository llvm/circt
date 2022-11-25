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

  // Verify the body.
  if (module->getNumRegions() != 1)
    return module.emitOpError("requires one region");

  // Verify the block arguments.
  auto &body = module->getRegion(0);
  if (!body.empty()) {
    auto &block = body.front();
    if (block.getNumArguments() != numPorts)
      return module.emitOpError("entry block must have ")
             << numPorts << " arguments to match module signature";

    if (llvm::any_of(
            llvm::zip(block.getArguments(), portTypes.getValue()),
            [](auto pair) {
              auto blockType = std::get<0>(pair).getType();
              auto portType =
                  std::get<1>(pair).template cast<TypeAttr>().getValue();
              return blockType != portType;
            }))
      return module.emitOpError(
          "block argument types should match signature types");
  }

  return success();
}

LogicalResult circt::firrtl::verifyInnerSymAttr(InnerSymbolOpInterface op) {
  auto innerSym = op.getInnerSymAttr();
  // If does not have any inner sym then ignore.
  if (!innerSym)
    return success();
  if (isa<InstanceOp>(&op) || op->getNumResults() != 1) {
    // The inner sym can only be specified on fieldID=0.
    if (innerSym.size() > 1 || !innerSym.getSymName()) {
      op->emitOpError("cannot assign symbols to non-zero field id, for ops "
                      "with zero or multiple results");
      return failure();
    }
    return success();
  }
  auto resultType = op->getResultTypes()[0].dyn_cast<FIRRTLBaseType>();
  if (!resultType)
    return op->emitOpError("cannot attach symbols to non-base types");
  auto maxFields = resultType.getMaxFieldID();
  SmallBitVector indices(maxFields + 1);
  SmallPtrSet<Attribute, 8> symNames;
  // Ensure fieldID and symbol names are unique.
  auto uniqSyms = [&](hw::InnerSymPropertiesAttr p) {
    if (maxFields < p.getFieldID()) {
      op->emitOpError("field id:'" + Twine(p.getFieldID()) +
                      "' is greater than the maximum field id:'" +
                      Twine(maxFields) + "'");
      return false;
    }
    if (indices.test(p.getFieldID())) {
      op->emitOpError("cannot assign multiple symbol names to the field id:'" +
                      Twine(p.getFieldID()) + "'");
      return false;
    }
    indices.set(p.getFieldID());
    auto it = symNames.insert(p.getName());
    if (!it.second) {
      op->emitOpError("cannot reuse symbol name:'" + p.getName().getValue() +
                      "'");
      return false;
    }
    return true;
  };

  if (!llvm::all_of(innerSym.getProps(), uniqSyms))
    return failure();
  return success();
}

#include "circt/Dialect/FIRRTL/FIRRTLOpInterfaces.cpp.inc"
