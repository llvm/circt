//===- RTLDialect.cpp - Implement the RTL dialect -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the RTL dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTL/RTLDialect.h"
#include "circt/Dialect/RTL/RTLOps.h"
#include "circt/Dialect/RTL/RTLTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

using namespace circt;
using namespace rtl;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

namespace {

// We implement the OpAsmDialectInterface so that RTL dialect operations
// automatically interpret the name attribute on operations as their SSA name.
struct RTLOpAsmDialectInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  /// Get a special name to use when printing the given operation. See
  /// OpAsmInterface.td#getAsmResultNames for usage details and documentation.
  void getAsmResultNames(Operation *op,
                         OpAsmSetValueNameFn setNameFn) const override {}

  /// Get a special name to use when printing the entry block arguments of the
  /// region contained by an operation in this dialect.
  void getAsmBlockArgumentNames(Block *block,
                                OpAsmSetValueNameFn setNameFn) const override {
    // Assign port names to the bbargs if this is a module.
    auto *parentOp = block->getParentOp();
    if (isAnyModule(parentOp)) {
      for (size_t i = 0, e = block->getNumArguments(); i != e; ++i) {
        auto name = getModuleArgumentName(parentOp, i);
        if (!name.empty())
          setNameFn(block->getArgument(i), name);
      }
    }
  }
};
} // end anonymous namespace

RTLDialect::RTLDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context,
              ::mlir::TypeID::get<RTLDialect>()) {

  registerTypes();

  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/RTL/RTL.cpp.inc"
      >();

  // Register interface implementations.
  addInterfaces<RTLOpAsmDialectInterface>();
}

RTLDialect::~RTLDialect() {}

// Registered hook to materialize a single constant operation from a given
/// attribute value with the desired resultant type. This method should use
/// the provided builder to create the operation without changing the
/// insertion position. The generated operation is expected to be constant
/// like, i.e. single result, zero operands, non side-effecting, etc. On
/// success, this hook should return the value generated to represent the
/// constant value. Otherwise, it should return null on failure.
Operation *RTLDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                           Type type, Location loc) {
  // Integer constants.
  if (auto intType = type.dyn_cast<IntegerType>())
    if (auto attrValue = value.dyn_cast<IntegerAttr>())
      return builder.create<rtl::ConstantOp>(loc, type, attrValue);

  return nullptr;
}

// Provide implementations for the enums we use.
#include "circt/Dialect/RTL/RTLEnums.cpp.inc"
