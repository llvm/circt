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

static Attribute parseRootedInstancePath(DialectAsmParser &p) {
  FlatSymbolRefAttr root;
  if (p.parseAttribute(root) || p.parseLSquare())
    return Attribute();
  SmallVector<StringAttr, 16> path;
  if (p.parseOptionalRSquare()) {
    do {
      StringAttr instName;
      if (p.parseAttribute(instName))
        return Attribute();
      path.push_back(instName);
    } while (!p.parseOptionalComma());
    if (p.parseRSquare())
      return Attribute();
  }
  return RootedInstancePathAttr::get(p.getContext(), root, path);
}

static void printRootedInstancePath(RootedInstancePathAttr me,
                                    DialectAsmPrinter &p) {
  p << me.getRootModule() << '[';
  llvm::interleave(me.getPath(), p, ",");
  p << ']';
}

Attribute SwitchInstanceAttr::parse(DialectAsmParser &p, Type type) {
  if (p.parseLess())
    return Attribute();
  if (!p.parseOptionalGreater())
    return SwitchInstanceAttr::get(p.getContext(), {});

  SmallVector<SwitchInstanceCaseAttr> instPairs;
  do {
    auto path =
        parseRootedInstancePath(p).dyn_cast_or_null<RootedInstancePathAttr>();
    if (!path)
      return Attribute();

    Attribute attr;
    if (p.parseEqual() || p.parseAttribute(attr))
      return Attribute();
    instPairs.push_back(
        SwitchInstanceCaseAttr::get(p.getContext(), path, attr));
  } while (!p.parseOptionalComma());
  if (p.parseGreater())
    return Attribute();

  return SwitchInstanceAttr::get(p.getContext(), instPairs);
}

void SwitchInstanceAttr::print(DialectAsmPrinter &p) const {
  p << "switch.inst<";
  llvm::interleaveComma(getCases(), p, [&](auto instCase) {
    printRootedInstancePath(instCase.getInst(), p);
    p << '=';
    p.printAttribute(instCase.getAttr());
  });
  p << '>';
}

Attribute SwitchInstanceAttr::lookup(RootedInstancePathAttr id) {
  // TODO: This is obviously very slow. Speed this up by using a sorted list.
  for (auto c : getCases())
    if (c.getInst() == id)
      return c.getAttr();
  return Attribute();
}

Attribute PhysLocationAttr::parse(DialectAsmParser &p, Type type) {
  llvm::SMLoc loc = p.getCurrentLocation();
  StringRef devTypeStr;
  uint64_t x, y, num;
  if (p.parseLess() || p.parseKeyword(&devTypeStr) || p.parseComma() ||
      p.parseInteger(x) || p.parseComma() || p.parseInteger(y) ||
      p.parseComma() || p.parseInteger(num) || p.parseGreater())
    return Attribute();

  Optional<PrimitiveType> devType = symbolizePrimitiveType(devTypeStr);
  if (!devType) {
    p.emitError(loc, "Unknown device type '" + devTypeStr + "'");
    return Attribute();
  }
  PrimitiveTypeAttr devTypeAttr =
      PrimitiveTypeAttr::get(p.getContext(), *devType);
  auto phy = PhysLocationAttr::get(p.getContext(), devTypeAttr, x, y, num);
  return phy;
}

void PhysLocationAttr::print(DialectAsmPrinter &p) const {
  p << "physloc<" << stringifyPrimitiveType(getPrimitiveType().getValue())
    << ", " << getX() << ", " << getY() << ", " << getNum() << '>';
}

Attribute LogicLockedRegionAttr::parse(DialectAsmParser &p, Type type) {
  StringRef regionName;
  uint64_t xMin, xMax, yMin, yMax;
  if (p.parseLess() || p.parseKeyword(&regionName) || p.parseComma() ||
      p.parseInteger(xMin) || p.parseComma() || p.parseInteger(xMax) ||
      p.parseComma() || p.parseInteger(yMin) || p.parseComma() ||
      p.parseInteger(yMax) || p.parseGreater()) {
    llvm::SMLoc loc = p.getCurrentLocation();
    p.emitError(loc, "unable to parse logic locked region");
    return Attribute();
  }

  auto regionNameAttr = StringAttr::get(p.getContext(), regionName);

  return LogicLockedRegionAttr::get(p.getContext(), regionNameAttr, xMin, xMax,
                                    yMin, yMax);
}

void LogicLockedRegionAttr::print(DialectAsmPrinter &p) const {
  p << "logic_locked_region<" << getRegionName().getValue() << ", " << getXMin()
    << ", " << getXMax() << ", " << getYMin() << ", " << getYMax() << '>';
}

Attribute MSFTDialect::parseAttribute(DialectAsmParser &p, Type type) const {
  StringRef attrName;
  Attribute attr;
  if (p.parseKeyword(&attrName))
    return Attribute();
  auto parseResult = generatedAttributeParser(p, attrName, type, attr);
  if (parseResult.hasValue())
    return attr;
  p.emitError(p.getNameLoc(), "Unexpected msft attribute '" + attrName + "'");
  return {};
}

void MSFTDialect::printAttribute(Attribute attr, DialectAsmPrinter &p) const {
  if (succeeded(generatedAttributePrinter(attr, p)))
    return;
  TypeSwitch<Attribute>(attr)
      .Case([&p](RootedInstancePathAttr path) {
        printRootedInstancePath(path, p);
      })
      .Default([](Attribute) { llvm_unreachable("Unexpected attribute"); });
}

void MSFTDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "circt/Dialect/MSFT/MSFTAttributes.cpp.inc"
      >();
}
