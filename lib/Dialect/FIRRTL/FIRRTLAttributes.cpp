//===- FIRRTLAttributes.cpp - Implement FIRRTL dialect attributes ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace firrtl;

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/FIRRTL/FIRRTLAttributes.cpp.inc"

//===----------------------------------------------------------------------===//
// PortDirectionsAttr
//===----------------------------------------------------------------------===//

llvm::raw_ostream &circt::firrtl::operator<<(llvm::raw_ostream &os,
                                             const Direction &dir) {
  return os << (dir == Direction::In ? "in" : "out");
}

Direction PortDirectionsAttr::operator[](unsigned index) {
  // This function exists to avoid copying the underlying array to query a bit.
  return getImpl()->value[index];
}

Attribute PortDirectionsAttr::parse(MLIRContext *ctxt, DialectAsmParser &p,
                                    Type type) {
  if (p.parseLess())
    return Attribute();
  PortDirections directions;
  while (true) {
    if (!p.parseOptionalKeyword("in"))
      directions.push_back(Direction::In);
    else if (!p.parseKeyword("out", "Expected 'in' or 'out'"))
      directions.push_back(Direction::Out);
    else
      return Attribute();
    // If there is no comma, break out of the loop.
    if (p.parseOptionalComma())
      break;
  }
  if (p.parseGreater())
    return Attribute();
  return PortDirectionsAttr::get(ctxt, directions);
}

void PortDirectionsAttr::print(DialectAsmPrinter &p) const {
  p << "directions<";
  llvm::interleaveComma(getValue(), p, [&](auto direction) {
    p << (direction == Direction::In ? "in" : "out");
  });
  p << '>';
}

//===----------------------------------------------------------------------===//
// InvalidValueAttr
//===----------------------------------------------------------------------===//

Attribute InvalidValueAttr::parse(MLIRContext *ctxt, DialectAsmParser &p,
                                  Type typeX) {
  FIRRTLType type;
  if (p.parseLess() || p.parseType(type) || p.parseGreater())
    return Attribute();
  return InvalidValueAttr::get(ctxt, type);
}

void InvalidValueAttr::print(DialectAsmPrinter &p) const {
  p << "invalidvalue<" << getType() << '>';
}

//===----------------------------------------------------------------------===//
// Dialect Attributes
//===----------------------------------------------------------------------===//

Attribute FIRRTLDialect::parseAttribute(DialectAsmParser &p, Type type) const {
  StringRef attrName;
  Attribute attr;
  if (p.parseKeyword(&attrName))
    return Attribute();
  auto parseResult =
      generatedAttributeParser(getContext(), p, attrName, type, attr);
  if (parseResult.hasValue())
    return attr;
  p.emitError(p.getNameLoc(), "Unexpected FIRRTL attribute '" + attrName + "'");
  return {};
}

void FIRRTLDialect::printAttribute(Attribute attr, DialectAsmPrinter &p) const {
  if (succeeded(generatedAttributePrinter(attr, p)))
    return;
  llvm_unreachable("Unexpected attribute");
}

//===----------------------------------------------------------------------===//
// SubAnnotationAttr
//===----------------------------------------------------------------------===//

Attribute SubAnnotationAttr::parse(MLIRContext *ctxt, DialectAsmParser &p,
                                   Type type) {
  int64_t fieldID;
  DictionaryAttr annotations;
  StringRef fieldIDKeyword;

  if (p.parseLess() || p.parseKeyword(&fieldIDKeyword) || p.parseEqual() ||
      p.parseInteger(fieldID) || p.parseComma() ||
      p.parseAttribute<DictionaryAttr>(annotations) || p.parseGreater())
    return Attribute();

  if (fieldIDKeyword != "fieldID")
    return Attribute();

  return SubAnnotationAttr::get(ctxt, fieldID, annotations);
}

void SubAnnotationAttr::print(DialectAsmPrinter &p) const {
  p << getMnemonic() << "<fieldID = " << getFieldID() << ", "
    << getAnnotations() << ">";
}

void FIRRTLDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "circt/Dialect/FIRRTL/FIRRTLAttributes.cpp.inc"
      >();
}
