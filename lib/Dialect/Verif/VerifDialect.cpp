//===- VerifDialect.cpp - Verif dialect implementation --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Verif/VerifDialect.h"
#include "circt/Dialect/Verif/VerifOps.h"

using namespace circt;
using namespace verif;

void VerifDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/Verif/Verif.cpp.inc"
      >();
}

#include "circt/Dialect/Verif/VerifDialect.cpp.inc"
