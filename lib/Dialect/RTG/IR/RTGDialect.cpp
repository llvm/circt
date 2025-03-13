//===- RTGDialect.cpp - Implement the RTG dialect -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the RTG dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTG/IR/RTGDialect.h"
#include "circt/Dialect/RTG/IR/RTGAttributes.h"
#include "circt/Dialect/RTG/IR/RTGOps.h"
#include "circt/Dialect/RTG/IR/RTGTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

using namespace circt;
using namespace rtg;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

void RTGDialect::initialize() {
  registerAttributes();
  registerTypes();
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/RTG/IR/RTG.cpp.inc"
      >();
}

/// Registered hook to materialize a single constant operation from a given
/// attribute value with the desired resultant type. This method should use
/// the provided builder to create the operation without changing the
/// insertion position. The generated operation is expected to be constant
/// like, i.e. single result, zero operands, non side-effecting, etc. On
/// success, this hook should return the value generated to represent the
/// constant value. Otherwise, it should return null on failure.
Operation *RTGDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                           Type type, Location loc) {
  if (auto attr = dyn_cast<ImmediateAttr>(value))
    if (type == attr.getType())
      return builder.create<ConstantOp>(loc, attr);

  return nullptr;
}

#include "circt/Dialect/RTG/IR/RTGEnums.cpp.inc"

#include "circt/Dialect/RTG/IR/RTGDialect.cpp.inc"
