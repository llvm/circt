//===- CombDialect.cpp - Implement the Comb dialect -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Comb dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

using namespace circt;
using namespace comb;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

void CombDialect::initialize() {
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/Comb/Comb.cpp.inc"
      >();
}

/// Registered hook to materialize a single constant operation from a given
/// attribute value with the desired resultant type. Reuse the materialization
/// from the HW dialect. See the documentation in HWDialect.cpp for more
/// information.
Operation *CombDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  return builder.getContext()
      ->getLoadedDialect<hw::HWDialect>()
      ->materializeConstant(builder, value, type, loc);
}

// Provide implementations for the enums we use.
#include "circt/Dialect/Comb/CombEnums.cpp.inc"

#include "circt/Dialect/Comb/CombDialect.cpp.inc"
