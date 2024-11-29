//===- RTGTestDialect.cpp - Implement the RTGTest dialect -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the RTGTest dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTGTest/IR/RTGTestDialect.h"
#include "circt/Dialect/RTGTest/IR/RTGTestOps.h"
#include "circt/Dialect/RTGTest/IR/RTGTestTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

using namespace circt;
using namespace rtgtest;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

void RTGTestDialect::initialize() {
  registerTypes();
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/RTGTest/IR/RTGTest.cpp.inc"
      >();
}

#include "circt/Dialect/RTGTest/IR/RTGTestEnums.cpp.inc"

#include "circt/Dialect/RTGTest/IR/RTGTestDialect.cpp.inc"
