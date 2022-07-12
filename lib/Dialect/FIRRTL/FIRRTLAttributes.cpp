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

//===----------------------------------------------------------------------===//
// Utilities related to Direction
//===----------------------------------------------------------------------===//

Direction direction::flip(Direction direction) {
  switch (direction) {
  case Direction::In:
    return Direction::Out;
  case Direction::Out:
    return Direction::In;
  }
  llvm_unreachable("unknown direction");
}

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

//===----------------------------------------------------------------------===//
// InnerSymAttr
//===----------------------------------------------------------------------===//

Attribute InnerSymPropertiesAttr::parse(AsmParser &parser, Type type) {
  StringAttr name;
  NamedAttrList dummyList;
  int64_t fieldId = 0;
  StringRef visibility;
  if (parser.parseLess() || parser.parseSymbolName(name, "name", dummyList) ||
      parser.parseComma() || parser.parseInteger(fieldId) ||
      parser.parseComma() ||
      parser.parseOptionalKeyword(&visibility,
                                  {"public", "private", "nested"}) ||
      parser.parseGreater())
    return Attribute();
  StringAttr visibilityAttr = parser.getBuilder().getStringAttr(visibility);

  return InnerSymPropertiesAttr::get(parser.getContext(), name, fieldId,
                                     visibilityAttr);
}

void InnerSymPropertiesAttr::print(AsmPrinter &p) const {
  p << "<@" << getName().getValue() << "," << getFieldID() << ","
    << getSymVisibility().getValue() << ">";
}

StringAttr InnerSymAttr::getSymIfExists(unsigned fieldId) {
  auto it =
      llvm::find_if(getImpl()->props, [&](const InnerSymPropertiesAttr &p) {
        return p.getFieldID() == fieldId;
      });
  if (it != getProps().end())
    return it->getName();
  return {};
}

StringAttr InnerSymAttr::getSymName() { return getSymIfExists(0); }

bool InnerSymAttr::all_of_props(
    std::function<bool(InnerSymPropertiesAttr)> func) {
  return llvm::all_of(getImpl()->props, func);
}

size_t InnerSymAttr::numSymbols() { return getImpl()->props.size(); }

Attribute InnerSymAttr::parse(AsmParser &parser, Type type) {
  StringAttr sym;
  NamedAttrList dummyList;
  SmallVector<InnerSymPropertiesAttr, 4> names;
  if (!parser.parseOptionalSymbolName(sym, "dummy", dummyList))
    names.push_back(InnerSymPropertiesAttr::get(sym));
  else if (parser.parseCommaSeparatedList(
               OpAsmParser::Delimiter::Square, [&]() -> ParseResult {
                 InnerSymPropertiesAttr prop;
                 if (parser.parseCustomAttributeWithFallback(
                         prop, mlir::Type{}, "dummy", dummyList))
                   return failure();

                 names.push_back(prop);

                 return success();
               }))
    return Attribute();

  std::sort(names.begin(), names.end(),
            [&](InnerSymPropertiesAttr a, InnerSymPropertiesAttr b) {
              return a.getFieldID() < b.getFieldID();
            });

  return InnerSymAttr::get(parser.getContext(), names);
}

void InnerSymAttr::print(AsmPrinter &p) const {

  auto props = getProps();
  if (props.size() == 1 &&
      props[0].getSymVisibility().getValue().equals("public") &&
      props[0].getFieldID() == 0) {
    p << "@" << props[0].getName().getValue();
    return;
  }
  auto names = props.vec();

  std::sort(names.begin(), names.end(),
            [&](InnerSymPropertiesAttr a, InnerSymPropertiesAttr b) {
              return a.getFieldID() < b.getFieldID();
            });
  p << "[";
  llvm::interleaveComma(names, p, [&](InnerSymPropertiesAttr attr) {
    attr.print(p);
    // p << "<@" << attr.getName().getValue() << "," << attr.getFieldID() << ","
    //   << attr.getSymVisibility().getValue() << ">";
  });
  p << "]";
}
