//===- PIRDialect.cpp - PIR dialect implementation ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/PIR/PIRDialect.h"
#include "circt/Dialect/PIR/PIROps.h"
#include "circt/Dialect/PIR/PIRTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"

using namespace circt;
using namespace pir;

void PIRDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/PIR/PIRTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/PIR/PIR.cpp.inc"
      >();
}

#include "circt/Dialect/PIR/PIRDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/PIR/PIRTypes.cpp.inc"
