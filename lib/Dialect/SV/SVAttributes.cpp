//===- SVAttributes.cpp - Implement the SV attributes ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SV/SVAttributes.h"
#include "circt/Dialect/SV/SVDialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace circt::sv;

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
  auto parseResult =
      generatedAttributeParser(getContext(), p, attrName, type, attr);
  if (parseResult.hasValue())
    return attr;
  p.emitError(p.getNameLoc(), "Unexpected sv attribute '" + attrName + "'");
  return {};
}

void SVDialect::printAttribute(Attribute attr, DialectAsmPrinter &p) const {
  if (succeeded(generatedAttributePrinter(attr, p)))
    return;
  llvm_unreachable("Unexpected attribute");
}

Attribute VerbatimParameterAttr::parse(MLIRContext *ctxt, DialectAsmParser &p,
                                       Type type) {
  StringRef verbatimString;
  if (p.parseLess() || p.parseOptionalString(&verbatimString) ||
      p.parseGreater())
    return Attribute();

  return VerbatimParameterAttr::get(ctxt, verbatimString);
}

void VerbatimParameterAttr::print(DialectAsmPrinter &p) const {
  p << "verbatim.parameter<\"";
  p.getStream().write_escaped(getValue());
  p << "\">";
}
