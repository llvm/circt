//===- CHALKDialect.cpp - Implementation of CHALK dialect ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/CHALK/CHALKDialect.h"
#include "circt/Dialect/CHALK/CHALKOps.h"
#include "mlir/IR/DialectImplementation.h"

using namespace circt;
using namespace chalk;

void CHALKDialect::initialize() {
  // Register types.
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/CHALK/CHALKTypes.cpp.inc"
      >();

  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/CHALK/CHALK.cpp.inc"
      >();
}
