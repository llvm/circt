//===- SMTDialect.cpp - SMT dialect implementation ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SMT/SMTDialect.h"

using namespace circt;
using namespace smt;

void SMTDialect::initialize() {
  registerTypes();
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/SMT/SMT.cpp.inc"
      >();
}

#include "circt/Dialect/SMT/SMTDialect.cpp.inc"
#include "circt/Dialect/SMT/SMTEnums.cpp.inc"
