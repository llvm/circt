//===- SeqDialect.cpp - Implement the Seq dialect -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Seq dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

using namespace circt;
using namespace seq;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

void SeqDialect::initialize() {
  registerTypes();
  registerAttributes();

  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/Seq/Seq.cpp.inc"
      >();
}

/// Registered hook to materialize a single constant operation from a given
/// attribute value with the desired resultant type. This method should use
/// the provided builder to create the operation without changing the
/// insertion position. The generated operation is expected to be constant
/// like, i.e. single result, zero operands, non side-effecting, etc. On
/// success, this hook should return the value generated to represent the
/// constant value. Otherwise, it should return null on failure.
Operation *SeqDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                           Type type, Location loc) {
  // Integer constants.
  if (auto intType = dyn_cast<IntegerType>(type))
    if (auto attrValue = dyn_cast<IntegerAttr>(value))
      return hw::ConstantOp::create(builder, loc, type, attrValue);

  if (isa<ClockType>(type))
    if (auto attrValue = dyn_cast<ClockConstAttr>(value))
      return seq::ConstClockOp::create(builder, loc, attrValue);

  return nullptr;
}

#include "circt/Dialect/Seq/SeqDialect.cpp.inc"
