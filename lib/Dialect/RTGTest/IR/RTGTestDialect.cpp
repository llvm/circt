//===- RTGTestDialect.cpp - Implement the RTGTest dialect -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the RTGTest dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTGTest/IR/RTGTestDialect.h"
#include "circt/Dialect/RTG/IR/RTGOps.h"
#include "circt/Dialect/RTGTest/IR/RTGTestOps.h"
#include "circt/Dialect/RTGTest/IR/RTGTestTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace rtgtest;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

void RTGTestDialect::initialize() {
  registerTypes();
  registerAttributes();
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/RTGTest/IR/RTGTest.cpp.inc"
      >();
}

/// Registered hook to materialize a single constant operation from a given
/// attribute value with the desired resultant type. This method should use
/// the provided builder to create the operation without changing the
/// insertion position. The generated operation is expected to be constant
/// like, i.e. single result, zero operands, non side-effecting, etc. On
/// success, this hook should return the value generated to represent the
/// constant value. Otherwise, it should return null on failure.
Operation *RTGTestDialect::materializeConstant(OpBuilder &builder,
                                               Attribute value, Type type,
                                               Location loc) {
  return TypeSwitch<Attribute, Operation *>(value)
      .Case<CPUAttr>([&](auto attr) -> Operation * {
        if (isa<CPUType>(type))
          return rtg::ConstantOp::create(builder, loc, attr);
        return nullptr;
      })
      .Case<rtg::RegisterAttrInterface>([&](auto attr) -> Operation * {
        if (isa<rtg::RegisterTypeInterface>(type))
          return rtg::FixedRegisterOp::create(builder, loc, attr);
        return nullptr;
      })
      .Default([](auto attr) { return nullptr; });
}

#include "circt/Dialect/RTGTest/IR/RTGTestEnums.cpp.inc"

#include "circt/Dialect/RTGTest/IR/RTGTestDialect.cpp.inc"
