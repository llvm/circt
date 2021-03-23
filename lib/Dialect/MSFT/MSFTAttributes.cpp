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
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace msft;

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/MSFT/MSFTAttributes.cpp.inc"

Attribute PhysLocationAttr::parse(MLIRContext *ctxt, DialectAsmParser &p,
                                  Type type) {
  llvm::SMLoc loc = p.getCurrentLocation();
  StringRef devTypeStr;
  uint64_t x, y, num;
  if (p.parseLess() || p.parseKeyword(&devTypeStr) || p.parseComma() ||
      p.parseInteger(x) || p.parseComma() || p.parseInteger(y) ||
      p.parseComma() || p.parseInteger(num) || p.parseGreater())
    return Attribute();

  Optional<DeviceType> devType = symbolizeDeviceType(devTypeStr);
  if (!devType) {
    p.emitError(loc, "Unknown device type '" + devTypeStr + "'");
    return Attribute();
  }
  DeviceTypeAttr devTypeAttr = DeviceTypeAttr::get(ctxt, *devType);
  auto phy = PhysLocationAttr::get(ctxt, devTypeAttr, x, y, num);
  return phy;
}

void PhysLocationAttr::print(DialectAsmPrinter &p) const {
  p << "physloc<" << stringifyDeviceType(getDevType().getValue()) << ", "
    << getX() << ", " << getY() << ", " << getNum() << '>';
}

Attribute MSFTDialect::parseAttribute(DialectAsmParser &p, Type type) const {
  StringRef attrName;
  Attribute attr;
  if (p.parseKeyword(&attrName))
    return Attribute();
  auto parseResult =
      generatedAttributeParser(getContext(), p, attrName, type, attr);
  if (parseResult.hasValue())
    return attr;
  p.emitError(p.getNameLoc(), "Unexpected msft attribute '" + attrName + "'");
  return {};
}

void MSFTDialect::printAttribute(Attribute attr, DialectAsmPrinter &p) const {
  if (succeeded(generatedAttributePrinter(attr, p)))
    return;
  llvm_unreachable("Unexpected attribute");
}

void MSFTDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "circt/Dialect/MSFT/MSFTAttributes.cpp.inc"
      >();
}
