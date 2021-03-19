//===- PhysicalDialect.cpp - Implement the Physical dialect ---------------===//
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

#include "circt/Dialect/Physical/PhysicalDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

using namespace circt;
using namespace physical;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

PhysicalDialect::PhysicalDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context,
              ::mlir::TypeID::get<PhysicalDialect>()) {}

PhysicalDialect::~PhysicalDialect() {}

/// Registered hook to materialize a single constant operation from a given
/// attribute value with the desired resultant type. This method should use
/// the provided builder to create the operation without changing the
/// insertion position. The generated operation is expected to be constant
/// like, i.e. single result, zero operands, non side-effecting, etc. On
/// success, this hook should return the value generated to represent the
/// constant value. Otherwise, it should return null on failure.
Operation *PhysicalDialect::materializeConstant(OpBuilder &builder,
                                                Attribute value, Type type,
                                                Location loc) {
  // Placeholder
  return nullptr;
}
