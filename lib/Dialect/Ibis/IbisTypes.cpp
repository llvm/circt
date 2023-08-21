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
  auto scopeRef = type.dyn_cast<ScopeRefType>();
  if (!scopeRef)
    return false;

  return scopeRef.isOpaque();
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
