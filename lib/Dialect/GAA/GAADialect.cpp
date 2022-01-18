//===- GAADialect.cpp - Implement the FIRRTL operations -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement the GAA Dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/GAA/GAADialect.h"
#include "circt/Dialect/GAA/GAAOps.h"

using namespace circt;
using namespace gaa;
StringRef InstanceOp::instanceName() { return getName(); }

void GAADialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/GAA/GAATypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/GAA/GAA.cpp.inc"
      >();
}

#include "circt/Dialect/GAA/GAADialect.cpp.inc"