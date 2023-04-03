//===- Dialect.cpp - Arc dialect implementation ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/Dialect.h"
#include "circt/Dialect/Arc/Ops.h"
#include "circt/Dialect/Arc/Types.h"

using namespace circt;
using namespace arc;

void ArcDialect::initialize() {
  registerTypes();
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/Arc/Arc.cpp.inc"
      >();
}

#include "circt/Dialect/Arc/ArcDialect.cpp.inc"
#include "circt/Dialect/Arc/ArcEnums.cpp.inc"
