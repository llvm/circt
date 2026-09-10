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

mlir::DenseBoolArrayAttr
direction::packAttribute(MLIRContext *context, ArrayRef<Direction> directions) {
  // Pack the array of directions into an APInt.  Input is zero, output is one.
  SmallVector<bool> dirs;
  dirs.reserve(directions.size());
  for (auto d : directions)
    dirs.push_back(d == Direction::Out);
  return mlir::DenseBoolArrayAttr::get(context, dirs);
}

mlir::DenseBoolArrayAttr direction::packAttribute(MLIRContext *context,
                                                  ArrayRef<bool> directions) {
  // Pack the array of directions into an APInt.  Input is zero, output is one.
  return mlir::DenseBoolArrayAttr::get(context, directions);
}

SmallVector<Direction>
direction::unpackAttribute(mlir::DenseBoolArrayAttr directions) {
  SmallVector<Direction> result;
  result.reserve(directions.size());
  for (auto d : directions.asArrayRef())
    result.push_back(direction::get(d));
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

Attribute ParamDeclAttr::parse(AsmParser &p, Type trailing) {
  std::string name;
  Type type;
  TypedAttr value;
  // < "FOO" : i32 > : i32
  // < "FOO" : i32 = 0 > : i32
  // < "FOO" : none >
  if (p.parseLess() || p.parseString(&name) || p.parseColonType(type))
    return Attribute();

  if (succeeded(p.parseOptionalEqual())) {
    if (p.parseAttribute(value, type))
      return Attribute();
  }

  if (p.parseGreater())
    return Attribute();

  if (value)
    return ParamDeclAttr::get(name, value);
  return ParamDeclAttr::get(name, type);
}

void ParamDeclAttr::print(AsmPrinter &p) const {
  p << "<" << getName() << ": " << getType();
  if (getValue()) {
    p << " = ";
    p.printAttributeWithoutType(getValue());
  }
  p << ">";
}
