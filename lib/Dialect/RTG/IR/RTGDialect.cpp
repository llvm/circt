//===- RTGDialect.cpp - Implement the RTG dialect -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the RTG dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTG/IR/RTGDialect.h"
#include "circt/Dialect/RTG/IR/RTGOps.h"
#include "circt/Dialect/RTG/IR/RTGTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

using namespace circt;
using namespace rtg;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

void RTGDialect::initialize() {
  registerTypes();
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/RTG/IR/RTG.cpp.inc"
      >();
}

#include "circt/Dialect/RTG/IR/RTGEnums.cpp.inc"

#include "circt/Dialect/RTG/IR/RTGDialect.cpp.inc"
