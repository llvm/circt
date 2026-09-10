//===- KanagawaTypes.cpp - Implementation of Kanagawa dialect types -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Kanagawa/KanagawaTypes.h"
#include "circt/Dialect/Kanagawa/KanagawaDialect.h"
#include "circt/Dialect/Kanagawa/KanagawaOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace kanagawa;

bool circt::kanagawa::isOpaqueScopeRefType(mlir::Type type) {
  auto scopeRef = dyn_cast<ScopeRefType>(type);
  if (!scopeRef)
    return false;

  return scopeRef.isOpaque();
}

Type ScopeRefType::parse(AsmParser &p) {
  if (p.parseOptionalLess())
    return ScopeRefType::get(p.getBuilder().getContext());

  SymbolRefAttr attr;
  if (p.parseAttribute(attr) || p.parseGreater())
    return Type();
  if (attr.getNestedReferences().size() != 1) {
    p.emitError(p.getNameLoc(), "expected @outer::@inner format");
    return Type();
  }
  return ScopeRefType::get(
      p.getBuilder().getContext(),
      hw::InnerRefAttr::get(attr.getRootReference(), attr.getLeafReference()));
}

void ScopeRefType::print(AsmPrinter &p) const {
  if (auto ref = getScopeRef())
    ref.print(p);
}

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/Kanagawa/KanagawaTypes.cpp.inc"

void KanagawaDialect::registerTypes() {
  // Register types.
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/Kanagawa/KanagawaTypes.cpp.inc"
      >();
}
