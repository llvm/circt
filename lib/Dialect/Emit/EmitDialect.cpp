//===- EmitDialect.cpp - Implement the Emit dialect -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Emit dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Emit/EmitDialect.h"
#include "circt/Dialect/Emit/EmitOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

using namespace circt;
using namespace emit;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

void EmitDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/Emit/Emit.cpp.inc"
      >();
}

#include "circt/Dialect/Emit/EmitDialect.cpp.inc"
