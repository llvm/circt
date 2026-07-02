//===- ICE40Dialect.cpp - Implement the ICE40 dialect ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the ICE40 dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ICE40/ICE40Dialect.h"
#include "circt/Dialect/ICE40/ICE40Ops.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

using namespace circt;
using namespace ice40;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

void ICE40Dialect::initialize() {
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/ICE40/ICE40.cpp.inc"
      >();
}

#include "circt/Dialect/ICE40/ICE40Dialect.cpp.inc"
