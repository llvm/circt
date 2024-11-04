//===- CalyxLibDialect.cpp - Implement the CalyxLib dialect ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the CalyxLib dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/CalyxLib/CalyxLibDialect.h"
#include "circt/Dialect/CalyxLib/CalyxLibOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

using namespace circt;
using namespace calyxlib;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

void CalyxLibDialect::initialize() {
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/CalyxLib/CalyxLib.cpp.inc"
      >();
  registerAttributes();
}

#include "circt/Dialect/CalyxLib/CalyxLibDialect.cpp.inc"
