//===- AIGDialect.cpp - Implement the AIG dialect -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AIG dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/AIG/AIGDialect.h"
#include "circt/Dialect/AIG/AIGOps.h"
#include "circt/Dialect/Synth/SynthDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

using namespace circt;
using namespace aig;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

void AIGDialect::initialize() {
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/AIG/AIG.cpp.inc"
      >();
}

#include "circt/Dialect/AIG/AIGDialect.cpp.inc"
