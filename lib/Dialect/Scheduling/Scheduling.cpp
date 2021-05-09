//===- Scheduling.cpp - Scheduling dialect implementation -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Scheduling dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Scheduling/Scheduling.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt::sched;
using mlir::Attribute;

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/Scheduling/SchedulingAttributes.cpp.inc"

void SchedulingDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "circt/Dialect/Scheduling/SchedulingAttributes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// OperatorInfo attribute
//===----------------------------------------------------------------------===//

Attribute OperatorInfoAttr::parse(MLIRContext *context, DialectAsmParser &p,
                                  Type type) {
  StringRef name;
  unsigned latency;
  if (p.parseLess() || p.parseOptionalString(&name) || p.parseComma() ||
      p.parseKeyword("latency") || p.parseEqual() || p.parseInteger(latency) ||
      p.parseGreater())
    return {};

  return OperatorInfoAttr::get(context, name, latency);
}

void OperatorInfoAttr::print(DialectAsmPrinter &p) const {
  p << getMnemonic() << "<\"" << getName() << "\", latency=" << getLatency()
    << '>';
}

//===----------------------------------------------------------------------===//
// Dispatching of dialect attribute parsing and printing
//===----------------------------------------------------------------------===//

Attribute SchedulingDialect::parseAttribute(DialectAsmParser &p,
                                            Type type) const {
  StringRef mnemonic;
  if (p.parseKeyword(&mnemonic))
    return {};

  Attribute attr;
  if (generatedAttributeParser(getContext(), p, mnemonic, type, attr)
          .hasValue())
    return attr;

  p.emitError(p.getNameLoc(), "Unexpected sched attribute '" + mnemonic + "'");
  return {};
}

void SchedulingDialect::printAttribute(Attribute attr,
                                       DialectAsmPrinter &p) const {
  if (succeeded(generatedAttributePrinter(attr, p)))
    return;
  llvm_unreachable("Unexpected attribute");
}
