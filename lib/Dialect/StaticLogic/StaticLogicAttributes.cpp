//===- StaticLogicAttributes.cpp - StaticLogic attributes -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/StaticLogic/StaticLogicAttributes.h"
#include "circt/Dialect/StaticLogic/StaticLogic.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt::staticlogic;

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/StaticLogic/StaticLogicAttributes.cpp.inc"

void StaticLogicDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "circt/Dialect/StaticLogic/StaticLogicAttributes.cpp.inc"
      >();
}

Attribute circt::staticlogic::InitiationIntervalAttr::parse(DialectAsmParser &p,
                                                            Type type) {
  int64_t ii;
  if (p.parseLess() || p.parseInteger(ii) || p.parseGreater()) {
    p.emitError(p.getNameLoc(), "unsupported II value");
    return Attribute();
  }

  auto *context = p.getContext();
  auto iiAttr = IntegerAttr::get(IntegerType::get(context, 64), ii);
  return InitiationIntervalAttr::get(context, iiAttr);
}

void circt::staticlogic::InitiationIntervalAttr::print(
    DialectAsmPrinter &p) const {
  p << "II<" << getValue().getInt() << '>';
}

Attribute StaticLogicDialect::parseAttribute(DialectAsmParser &p,
                                             Type type) const {
  StringRef attrName;
  Attribute attr;
  if (p.parseKeyword(&attrName))
    return Attribute();
  auto parseResult = generatedAttributeParser(p, attrName, type, attr);
  if (parseResult.hasValue())
    return attr;
  p.emitError(p.getNameLoc(),
              "Unexpected StaticLogic attribute '" + attrName + "'");
  return Attribute();
}

void StaticLogicDialect::printAttribute(Attribute attr,
                                        DialectAsmPrinter &p) const {
  if (succeeded(generatedAttributePrinter(attr, p)))
    return;
  llvm_unreachable("Unexpected attribute");
}
