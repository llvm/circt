//===- IbisTypes.cpp - Implementation of Ibis dialect types ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Ibis/IbisTypes.h"
#include "circt/Dialect/Ibis/IbisDialect.h"
#include "circt/Dialect/Ibis/IbisOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace ibis;

bool circt::ibis::isOpaqueScopeRefType(mlir::Type type) {
  auto scopeRef = dyn_cast<ScopeRefType>(type);
  if (!scopeRef)
    return false;

  return scopeRef.isOpaque();
}

Type ScopeRefType::parse(AsmParser &p) {
  // Is this an opaque scoperef? (!ibis.scoperef => no innerref)
  if (p.parseOptionalLess())
    return ScopeRefType::get(p.getBuilder().getContext());

  // Else, parse the inner reference
  hw::InnerRefAttr innerRefAttr = hw::parseInnerRefAttr(p, /*parseLess=*/false)
                                      .dyn_cast_or_null<hw::InnerRefAttr>();

  if (p.parseGreater() || !innerRefAttr)
    return Type();

  return ScopeRefType::get(p.getBuilder().getContext(), innerRefAttr);
}

void ScopeRefType::print(AsmPrinter &p) const {
  if (auto ref = getScopeRef())
    ref.print(p);
}

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/Ibis/IbisTypes.cpp.inc"

void IbisDialect::registerTypes() {
  // Register types.
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/Ibis/IbisTypes.cpp.inc"
      >();
}
