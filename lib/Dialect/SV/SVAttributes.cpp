//===- SVAttributes.cpp - Implement SV attributes -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SV/SVAttributes.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/SV/SVTypes.h"
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

#include "circt/Dialect/SV/SVEnums.cpp.inc"

bool circt::sv::hasSVAttributes(mlir::Operation *op) {
  return op->hasAttr(sv::SVAttributeAttr::getSVAttributesAttrName());
}

SVAttributesAttr circt::sv::getSVAttributes(mlir::Operation *op) {
  return op->getAttrOfType<SVAttributesAttr>(
      SVAttributeAttr::getSVAttributesAttrName());
}

void circt::sv::setSVAttributes(mlir::Operation *op, mlir::Attribute attr) {
  return op->setAttr(SVAttributeAttr::getSVAttributesAttrName(), attr);
}

mlir::Attribute SVAttributesAttr::parse(mlir::AsmParser &p, mlir::Type type) {
  mlir::ArrayAttr attributes;
  if (p.parseLess() || p.parseAttribute<ArrayAttr>(attributes))
    return Attribute();
  bool emitAsComments = false;
  if (!p.parseOptionalComma()) {
    if (p.parseKeyword("emitAsComments"))
      return Attribute();
    emitAsComments = true;
  }

  if (p.parseGreater())
    return Attribute();

  return SVAttributesAttr::get(p.getContext(), attributes,
                               BoolAttr::get(p.getContext(), emitAsComments));
}

SVAttributesAttr
SVAttributesAttr::get(MLIRContext *context,
                      ArrayRef<std::pair<StringRef, StringRef>> keyValuePairs,
                      bool emitAsComments) {
  SmallVector<Attribute> attrs;
  for (auto [key, value] : keyValuePairs)
    attrs.push_back(sv::SVAttributeAttr::get(
        context, StringAttr::get(context, key),
        value.empty() ? StringAttr() : StringAttr::get(context, value)));
  return SVAttributesAttr::get(context, ArrayAttr::get(context, attrs),
                               BoolAttr::get(context, emitAsComments));
}

void SVAttributesAttr::print(::mlir::AsmPrinter &p) const {
  p << "<" << getAttributes();
  if (getEmitAsComments().getValue())
    p << ", emitAsComments";
  p << ">";
}

void SVDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "circt/Dialect/SV/SVAttributes.cpp.inc"
      >();
}
