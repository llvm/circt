//===- APIntDialect.cpp - Implement the APInt dialect ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the APInt dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/APInt/APIntDialect.h"
#include "circt/Dialect/APInt/APIntOps.h"

using namespace circt;
using namespace apint;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

void APIntDialect::initialize() {
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/APInt/APInt.cpp.inc"
      >();
}

#include "circt/Dialect/APInt/APIntDialect.cpp.inc"
