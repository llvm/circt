//===- DCDialect.cpp - DC dialect implementation --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/DC/DCDialect.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace circt;
using namespace dc;

void DCDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/DC/DC.cpp.inc"
      >();
}

#include "circt/Dialect/DC/DCDialect.cpp.inc"
