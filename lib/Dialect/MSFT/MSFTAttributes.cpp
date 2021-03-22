//===- MSFTAttributescpp - Implement MSFT dialect attributes --------------===//
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
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace msft;

static Attribute parsePhysLocation(DialectAsmParser &p) {
  llvm::SMLoc loc = p.getCurrentLocation();
  StringRef devTypeStr;
  uint64_t x, y, num;
  StringAttr entity;
  if (p.parseLess() || p.parseKeyword(&devTypeStr) || p.parseComma() ||
      p.parseInteger(x) || p.parseComma() || p.parseInteger(y) ||
      p.parseComma() || p.parseInteger(num) || p.parseComma() ||
      p.parseAttribute(entity) || p.parseGreater())
    return Attribute();

  auto *ctxt = p.getBuilder().getContext();
  Optional<DeviceType> devType = symbolizeDeviceType(devTypeStr);
  if (!devType) {
    p.emitError(loc, "Unknown device type '" + devTypeStr + "'");
    return Attribute();
  }
  DeviceTypeAttr devTypeAttr = DeviceTypeAttr::get(ctxt, *devType);
  auto phy = PhysLocationAttr::get(p.getBuilder().getContext(), devTypeAttr, x,
                                   y, num, entity.getValue());
  return phy;
}

static void print(DialectAsmPrinter &p, PhysLocationAttr a) {
  p << "physloc<" << stringifyDeviceType(a.getDevType().getValue()) << ", "
    << a.getX() << ", " << a.getY() << ", " << a.getNum() << ", \""
    << a.getEntity() << "\">";
}

Attribute MSFTDialect::parseAttribute(DialectAsmParser &p, Type type) const {
  StringRef attrName;
  if (p.parseKeyword(&attrName))
    return Attribute();
  if (attrName == "physloc")
    return parsePhysLocation(p);
  p.emitError(p.getNameLoc(), "Unexpected msft attribute '" + attrName + "'");
  return Attribute();
}

void MSFTDialect::printAttribute(Attribute attr, DialectAsmPrinter &p) const {
  TypeSwitch<Attribute>(attr)
      .Case([&p](PhysLocationAttr a) { print(p, a); })
      .Default([](Attribute) { llvm_unreachable("Unexpected attribute"); });
}

void MSFTDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "circt/Dialect/MSFT/MSFTAttributes.cpp.inc"
      >();
}

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/MSFT/MSFTAttributes.cpp.inc"
