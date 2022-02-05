//===- MSFTAttributes.cpp - Implement MSFT dialect attributes -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the MSFT dialect attributes.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/MSFT/MSFTAttributes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace msft;

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/MSFT/MSFTAttributes.cpp.inc"

Attribute PhysLocationAttr::parse(AsmParser &p, Type type) {
  llvm::SMLoc loc = p.getCurrentLocation();
  std::string subPath;
  StringRef devTypeStr;
  uint64_t x, y, num;

  if (p.parseLess() || p.parseKeyword(&devTypeStr) || p.parseComma() ||
      p.parseInteger(x) || p.parseComma() || p.parseInteger(y) ||
      p.parseComma() || p.parseInteger(num))
    return Attribute();

  // Parse an optional subPath.
  if (succeeded(p.parseOptionalComma()))
    if (p.parseString(&subPath))
      return Attribute();

  if (p.parseGreater())
    return Attribute();

  Optional<PrimitiveType> devType = symbolizePrimitiveType(devTypeStr);
  if (!devType) {
    p.emitError(loc, "Unknown device type '" + devTypeStr + "'");
    return Attribute();
  }
  PrimitiveTypeAttr devTypeAttr =
      PrimitiveTypeAttr::get(p.getContext(), *devType);
  auto phy =
      PhysLocationAttr::get(p.getContext(), devTypeAttr, x, y, num, subPath);
  return phy;
}

void PhysLocationAttr::print(AsmPrinter &p) const {
  p << "<" << stringifyPrimitiveType(getPrimitiveType().getValue()) << ", "
    << getX() << ", " << getY() << ", " << getNum();

  // Print an optional subPath.
  if (!getSubPath().empty())
    p << ", \"" << getSubPath() << '"';

  p << '>';
}

Attribute PhysicalRegionRefAttr::parse(AsmParser &p, Type type) {
  StringAttr physicalRegion;
  NamedAttrList attrs;
  if (p.parseLess() ||
      p.parseSymbolName(physicalRegion, "physicalRegion", attrs) ||
      p.parseGreater()) {
    llvm::SMLoc loc = p.getCurrentLocation();
    p.emitError(loc, "unable to parse PhysicalRegion reference");
    return Attribute();
  }

  auto physicalRegionAttr =
      FlatSymbolRefAttr::get(p.getContext(), physicalRegion.getValue());

  return PhysicalRegionRefAttr::get(p.getContext(), physicalRegionAttr);
}

void PhysicalRegionRefAttr::print(AsmPrinter &p) const {
  p << "<" << getPhysicalRegion() << '>';
}

Attribute PhysicalBoundsAttr::parse(AsmParser &p, Type type) {
  uint64_t xMin, xMax, yMin, yMax;
  if (p.parseLess() || p.parseKeyword("x") || p.parseColon() ||
      p.parseLSquare() || p.parseInteger(xMin) || p.parseComma() ||
      p.parseInteger(xMax) || p.parseRSquare() || p.parseComma() ||
      p.parseKeyword("y") || p.parseColon() || p.parseLSquare() ||
      p.parseInteger(yMin) || p.parseComma() || p.parseInteger(yMax) ||
      p.parseRSquare() || p.parseGreater()) {
    llvm::SMLoc loc = p.getCurrentLocation();
    p.emitError(loc, "unable to parse PhysicalBounds");
    return Attribute();
  }

  return PhysicalBoundsAttr::get(p.getContext(), xMin, xMax, yMin, yMax);
}

void PhysicalBoundsAttr::print(AsmPrinter &p) const {
  p << "<";
  p << "x: [" << getXMin() << ", " << getXMax() << "], ";
  p << "y: [" << getYMin() << ", " << getYMax() << ']';
  p << '>';
}

void MSFTDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "circt/Dialect/MSFT/MSFTAttributes.cpp.inc"
      >();
}
