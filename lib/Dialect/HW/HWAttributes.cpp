//===- HWAttributes.cpp - Implement HW attributes -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace circt::hw;

//===----------------------------------------------------------------------===//
// ODS Boilerplate
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/HW/HWAttributes.cpp.inc"

void HWDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "circt/Dialect/HW/HWAttributes.cpp.inc"
      >();
}

Attribute HWDialect::parseAttribute(DialectAsmParser &p, Type type) const {
  StringRef attrName;
  Attribute attr;
  if (p.parseKeyword(&attrName))
    return Attribute();
  auto parseResult =
      generatedAttributeParser(getContext(), p, attrName, type, attr);
  if (parseResult.hasValue())
    return attr;
  p.emitError(p.getNameLoc(), "Unexpected hw attribute '" + attrName + "'");
  return {};
}

void HWDialect::printAttribute(Attribute attr, DialectAsmPrinter &p) const {
  if (succeeded(generatedAttributePrinter(attr, p)))
    return;
  llvm_unreachable("Unexpected attribute");
}

//===----------------------------------------------------------------------===//
// ParamDeclAttr
//===----------------------------------------------------------------------===//

Attribute ParamDeclAttr::parse(MLIRContext *context, DialectAsmParser &p,
                               Type type) {
  llvm::errs() << "Should never parse raw\n";
  abort();
}

void ParamDeclAttr::print(DialectAsmPrinter &p) const {
  llvm::errs() << "Should never print raw\n";
  abort();
}

//===----------------------------------------------------------------------===//
// ParamDeclRefAttr
//===----------------------------------------------------------------------===//

Attribute ParamDeclRefAttr::parse(MLIRContext *context, DialectAsmParser &p,
                                  Type type) {
  StringAttr name;
  if (p.parseLess() || p.parseAttribute(name) || p.parseGreater())
    return Attribute();

  return ParamDeclRefAttr::get(context, name, type);
}

void ParamDeclRefAttr::print(DialectAsmPrinter &p) const {
  p << "param.decl.ref<" << getName() << ">";
}

//===----------------------------------------------------------------------===//
// ParamVerbatimAttr
//===----------------------------------------------------------------------===//

Attribute ParamVerbatimAttr::parse(MLIRContext *context, DialectAsmParser &p,
                                   Type type) {
  StringAttr text;
  if (p.parseLess() || p.parseAttribute(text) || p.parseGreater())
    return Attribute();

  return ParamVerbatimAttr::get(context, text, type);
}

void ParamVerbatimAttr::print(DialectAsmPrinter &p) const {
  p << "param.verbatim<" << getValue() << ">";
}

//===----------------------------------------------------------------------===//
// ParamBinaryAttr
//===----------------------------------------------------------------------===//

Attribute ParamBinaryAttr::parse(MLIRContext *context, DialectAsmParser &p,
                                 Type type) {
  Attribute lhs, rhs;
  StringRef opcodeStr;
  auto loc = p.getCurrentLocation();
  if (p.parseLess() || p.parseKeyword(&opcodeStr) ||
      p.parseAttribute(lhs, type) || p.parseComma() ||
      p.parseAttribute(rhs, type) || p.parseGreater())
    return Attribute();

  Optional<PBO> opcode = symbolizePBO(opcodeStr);
  if (!opcode.hasValue()) {
    p.emitError(loc, "unknown binary operator name");
    return {};
  }

  return ParamBinaryAttr::get(context, *opcode, lhs, rhs, type);
}

void ParamBinaryAttr::print(DialectAsmPrinter &p) const {
  p << "param.binary<" << stringifyPBO(getOpcode()) << " ";
  p.printAttributeWithoutType(getLhs());
  p << ", ";
  p.printAttributeWithoutType(getRhs());
  p << ">";
}
