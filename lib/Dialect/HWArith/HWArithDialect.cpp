//===- HWArithDialect.cpp - Implement the HWArith dialect -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the HWArith dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HWArith/HWArithDialect.h"
#include "circt/Dialect/HWArith/HWArithOps.h"
#include "mlir/IR/DialectImplementation.h"

using namespace circt;
using namespace circt::hwarith;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

void HWArithDialect::initialize() {
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/HWArith/HWArith.cpp.inc"
      >();
}

// Provide implementations for the enums we use.
#include "circt/Dialect/HWArith/HWArithEnums.cpp.inc"

#include "circt/Dialect/HWArith/HWArithDialect.cpp.inc"
