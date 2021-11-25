//===- APIntTypes.cpp - HW types code defs -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation logic for APInt data types.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/APInt/APIntTypes.h"

using namespace circt;
using namespace circt::apint;

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/APInt/APIntTypes.cpp.inc"

/// Parses a type registered to this dialect
Type APIntDialect::parseType(DialectAsmParser &parser) const {
  return Type();
}

/// Print a type registered to this dialect
void APIntDialect::printType(Type type, DialectAsmPrinter &printer) const {
  llvm_unreachable("unexpected 'apint' type kind");
}

void APIntDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/APInt/APIntTypes.cpp.inc"
      >();
}
