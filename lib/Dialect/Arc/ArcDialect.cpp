//===- ArcDialect.cpp - Arc dialect implementation ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcDialect.h"
#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcTypes.h"

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
