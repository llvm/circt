//===- BLIFDialect.cpp - Implement the BLIF dialect -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the BLIF dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/BLIF/BLIFDialect.h"
#include "circt/Dialect/BLIF/BLIFOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

using namespace circt;
using namespace blif;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

void BLIFDialect::initialize() {
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/BLIF/BLIF.cpp.inc"
      >();
}

// Provide implementations for the enums we use.
#include "circt/Dialect/BLIF/BLIFEnums.cpp.inc"

#include "circt/Dialect/BLIF/BLIFDialect.cpp.inc"
