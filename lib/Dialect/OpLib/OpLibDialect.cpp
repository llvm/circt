//===- OpLibDialect.cpp - Implement the OpLib dialect ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the OpLib dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/OpLib/OpLibDialect.h"
#include "circt/Dialect/OpLib/OpLibOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

using namespace circt;
using namespace oplib;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

void OpLibDialect::initialize() {
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/OpLib/OpLib.cpp.inc"
      >();
  registerAttributes();
}

#include "circt/Dialect/OpLib/OpLibDialect.cpp.inc"
