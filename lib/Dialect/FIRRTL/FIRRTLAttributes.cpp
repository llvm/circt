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
#include "mlir/IR/SymbolTable.h"
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

Attribute InnerSymAttr::parse(AsmParser &p, Type type) {
  //  A sample IR, parse begins after `sym`.
  //  %wire = firrtl.wire sym @wireSym<fieldID=1><sym_visibility="private"> :
  StringAttr sym;
  NamedAttrList dummyList;
  if (p.parseSymbolName(sym, "dummy", dummyList))
    return Attribute();
  return InnerSymAttr::get(p.getContext(), sym);
}

void InnerSymAttr::print(AsmPrinter &p) const {
  //  A sample IR, print begins after `sym`.
  //  %wire = firrtl.wire sym @wireSym<fieldID=1><sym_visibility="private"> :

  p << "@" << getSymName().getValue();
}
