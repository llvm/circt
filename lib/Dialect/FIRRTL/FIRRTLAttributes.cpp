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
#include <iterator>

using namespace circt;
using namespace firrtl;

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/FIRRTL/FIRRTLAttributes.cpp.inc"

Attribute InvalidValueAttr::parse(AsmParser &p, Type typeX) {
  FIRRTLType type;
  if (p.parseLess() || p.parseType(type) || p.parseGreater())
    return Attribute();
  return InvalidValueAttr::get(p.getContext(), type);
}

void InvalidValueAttr::print(AsmPrinter &p) const {
  p << '<' << getType() << '>';
}

//===----------------------------------------------------------------------===//
// SubAnnotationAttr
//===----------------------------------------------------------------------===//

Attribute SubAnnotationAttr::parse(AsmParser &p, Type type) {
  int64_t fieldID;
  DictionaryAttr annotations;
  StringRef fieldIDKeyword;

  if (p.parseLess() || p.parseKeyword(&fieldIDKeyword) || p.parseEqual() ||
      p.parseInteger(fieldID) || p.parseComma() ||
      p.parseAttribute<DictionaryAttr>(annotations) || p.parseGreater())
    return Attribute();

  if (fieldIDKeyword != "fieldID")
    return Attribute();

  return SubAnnotationAttr::get(p.getContext(), fieldID, annotations);
}

void SubAnnotationAttr::print(AsmPrinter &p) const {
  p << "<fieldID = " << getFieldID() << ", " << getAnnotations() << ">";
}

//===----------------------------------------------------------------------===//
// Utilities related to Direction
//===----------------------------------------------------------------------===//

IntegerAttr direction::packAttribute(MLIRContext *context,
                                     ArrayRef<Direction> directions) {
  // Pack the array of directions into an APInt.  Input is zero, output is one.
  auto size = directions.size();
  APInt portDirections(size, 0);
  for (size_t i = 0; i != size; ++i)
    if (directions[i] == Direction::Out)
      portDirections.setBit(i);
  return IntegerAttr::get(IntegerType::get(context, size), portDirections);
}

SmallVector<Direction> direction::unpackAttribute(IntegerAttr directions) {
  assert(directions.getType().isSignlessInteger() &&
         "Direction attributes must be signless integers");
  auto value = directions.getValue();
  auto size = value.getBitWidth();
  SmallVector<Direction> result;
  result.reserve(size);
  for (size_t i = 0; i != size; ++i)
    result.push_back(direction::get(value[i]));
  return result;
}

llvm::raw_ostream &circt::firrtl::operator<<(llvm::raw_ostream &os,
                                             const Direction &dir) {
  return os << direction::toString(dir);
}

void FIRRTLDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "circt/Dialect/FIRRTL/FIRRTLAttributes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ParamDeclAttr
//===----------------------------------------------------------------------===//

Attribute ParamDeclAttr::parse(AsmParser &p, Type type) {
  llvm::errs() << "Should never parse raw\n";
  abort();
}

void ParamDeclAttr::print(AsmPrinter &p) const {
  p << "<" << getName() << ": " << getType();
  if (getValue())
    p << " = " << getValue();
  p << ">";
}
