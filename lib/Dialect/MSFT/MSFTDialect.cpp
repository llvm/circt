//===- MSFTDialect.cpp - Implement the MSFT dialect -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the MSFT dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/MSFT/MSFTDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace msft;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

MSFTDialect::MSFTDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context,
              ::mlir::TypeID::get<MSFTDialect>()) {}

MSFTDialect::~MSFTDialect() {}

/// Registered hook to materialize a single constant operation from a given
/// attribute value with the desired resultant type. This method should use
/// the provided builder to create the operation without changing the
/// insertion position. The generated operation is expected to be constant
/// like, i.e. single result, zero operands, non side-effecting, etc. On
/// success, this hook should return the value generated to represent the
/// constant value. Otherwise, it should return null on failure.
Operation *MSFTDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  // Placeholder
  return nullptr;
}

Attribute parsePhysLocation(DialectAsmParser &p, Type type) {
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
  auto ui64 = p.getBuilder().getIntegerType(64, false);
  auto phy = PhysLocationAttr::get(devTypeAttr, IntegerAttr::get(ui64, x),
                                   IntegerAttr::get(ui64, y),
                                   IntegerAttr::get(ui64, num), entity, ctxt);
  return phy;
}

void print(DialectAsmPrinter &p, PhysLocationAttr a) {
  p << "msft.physloc<" << a.Type() << ',' << a.X() << ',' << a.Y() << ','
    << a.Num() << ',' << a.Entity() << '>';
}

Attribute MSFTDialect::parseAttribute(DialectAsmParser &p, Type type) const {
  StringRef attrName;
  if (p.parseKeyword(&attrName))
    return Attribute();
  if (attrName == "physloc")
    return parsePhysLocation(p, type);
  llvm_unreachable("Unexpected 'msft' attribute");
}

void MSFTDialect::printAttribute(Attribute attr, DialectAsmPrinter &p) const {
  TypeSwitch<Attribute>(attr)
      .Case([&p](PhysLocationAttr a) { print(p, a); })
      .Default([](Attribute) { llvm_unreachable("Unexpected attribute"); });
}

#include "circt/Dialect/MSFT/MSFTAttrs.cpp.inc"
#include "circt/Dialect/MSFT/MSFTEnums.cpp.inc"
