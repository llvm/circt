//===- SystemCAttributes.cpp - SystemC attribute code defs ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation for SystemC attributes.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SystemC/SystemCAttributes.h"
#include "circt/Dialect/SystemC/SystemCDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt::systemc;
using namespace mlir;

//===----------------------------------------------------------------------===//
// PortDirectionsAttr
//===----------------------------------------------------------------------===//

PortDirection PortDirectionsAttr::getDirection(unsigned idx) const {
  return symbolizePortDirection(
             getStorage().getValue().extractBitsAsZExtValue(2, 2 * idx))
      .value();
}

void PortDirectionsAttr::getPortDirections(
    SmallVectorImpl<PortDirection> &directions) const {
  for (unsigned i = 0, e = getStorage().getValue().getBitWidth() >> 1; i < e;
       ++i) {
    directions.push_back(getDirection(i));
  }
}

PortDirectionsAttr PortDirectionsAttr::get(MLIRContext *context,
                                           ArrayRef<PortDirection> directions) {
  APInt storage(2 * directions.size(), 0);
  for (size_t i = 0; i < directions.size(); ++i) {
    storage.insertBits(APInt(2, static_cast<uint32_t>(directions[i])), 2 * i);
  }
  return get(context, storage);
}

Attribute PortDirectionsAttr::parse(AsmParser &p, Type odsType) {
  if (p.parseLess())
    return {};

  SmallVector<PortDirection> directions;

  auto parsePortDirection = [&]() -> ParseResult {
    if (succeeded(p.parseOptionalKeyword(
            stringifyPortDirection(PortDirection::InOut)))) {
      directions.push_back(PortDirection::InOut);
      return success();
    }
    if (succeeded(p.parseOptionalKeyword(
            stringifyPortDirection(PortDirection::Input)))) {
      directions.push_back(PortDirection::Input);
      return success();
    }
    if (succeeded(p.parseOptionalKeyword(
            stringifyPortDirection(PortDirection::Output)))) {
      directions.push_back(PortDirection::Output);
      return success();
    }
    return failure();
  };

  if (p.parseCommaSeparatedList(OpAsmParser::Delimiter::Square,
                                parsePortDirection))
    return {};

  if (p.parseGreater())
    return {};

  return PortDirectionsAttr::get(p.getContext(), directions);
}

void PortDirectionsAttr::print(AsmPrinter &p) const {
  p << "<[";
  for (size_t i = 0, e = getNumPorts(); i < e; ++i) {
    if (i > 0)
      p << ", ";

    p << stringifyPortDirection(getDirection(i));
  }
  p << "]>";
}

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/SystemC/SystemCAttributes.cpp.inc"
#include "circt/Dialect/SystemC/SystemCEnums.cpp.inc"

void SystemCDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "circt/Dialect/SystemC/SystemCAttributes.cpp.inc"
      >();
}
