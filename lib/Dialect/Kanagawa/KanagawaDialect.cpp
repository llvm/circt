//===- KanagawaDialect.cpp - Implementation of Kanagawa dialect -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Kanagawa/KanagawaDialect.h"
#include "circt/Dialect/Kanagawa/KanagawaOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace kanagawa;

// Pull in the dialect definition.
#include "circt/Dialect/Kanagawa/KanagawaDialect.cpp.inc"

void KanagawaDialect::initialize() {
  registerTypes();
  registerAttributes();

  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/Kanagawa/Kanagawa.cpp.inc"
      >();
}

void KanagawaDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "circt/Dialect/Kanagawa/KanagawaAttributes.cpp.inc"
      >();
}

// Provide implementations for the enums we use.
#include "circt/Dialect/Kanagawa/KanagawaEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/Kanagawa/KanagawaAttributes.cpp.inc"
