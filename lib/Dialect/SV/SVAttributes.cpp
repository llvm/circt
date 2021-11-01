//===- SVAttributes.cpp - Implement SV Attributes--------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides implementations for SV attributes.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SV/SVAttributes.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace circt::sv;

//===----------------------------------------------------------------------===//
// ODS Boilerplate
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/SV/SVAttributes.cpp.inc"

void SVDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "circt/Dialect/SV/SVAttributes.cpp.inc"
      >();
}

Attribute SVDialect::parseAttribute(DialectAsmParser &p, Type type) const {
  StringRef attrName;
  Attribute attr;
  if (p.parseKeyword(&attrName))
    return Attribute();
  auto parseResult = generatedAttributeParser(p, attrName, type, attr);
  if (parseResult.hasValue())
    return attr;
  p.emitError(p.getNameLoc(), "Unexpected SV attribute '" + attrName + "'");
  return {};
}

void SVDialect::printAttribute(Attribute attr, DialectAsmPrinter &p) const {
  if (succeeded(generatedAttributePrinter(attr, p)))
    return;
  llvm_unreachable("Unexpected attribute");
}

//===----------------------------------------------------------------------===//
// AttributeInstanceAttr
//===----------------------------------------------------------------------===//

Attribute SVAttributeAttr::parse(DialectAsmParser &p, Type type) {
  StringAttr name, expression;
  if (p.parseLess() || p.parseAttribute<StringAttr>(name))
    return Attribute();
  p.parseOptionalEqual() || p.parseAttribute(expression);
  if (p.parseGreater())
    return Attribute();
  return SVAttributeAttr::get(p.getContext(), name, expression);
}

void SVAttributeAttr::print(::mlir::DialectAsmPrinter &p) const {
  p << getMnemonic() << "<" << getName();
  if (auto expression = getExpression())
    p << "=" << expression;
  p << ">";
}
