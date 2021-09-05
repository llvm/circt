//===- CalyxDialect.cpp - Implement the Calyx dialect ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Calyx dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Calyx/CalyxDialect.h"
#include "circt/Dialect/Calyx/CalyxOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

using namespace circt;
using namespace circt::calyx;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

namespace {

// We implement the OpAsmDialectInterface so that Calyx dialect operations
// automatically interpret the name attribute on operations as their SSA name.
struct CalyxOpAsmDialectInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  /// Get a special name to use when printing the given operation. See
  /// OpAsmInterface.td#getAsmResultNames for usage details and documentation.
  void getAsmResultNames(Operation *op,
                         OpAsmSetValueNameFn setNameFn) const override {}

  /// Get a special name to use when printing the entry block arguments of the
  /// region contained by an operation in this dialect.
  void getAsmBlockArgumentNames(Block *block,
                                OpAsmSetValueNameFn setNameFn) const override;
};

} // end anonymous namespace

void CalyxOpAsmDialectInterface::getAsmBlockArgumentNames(
    Block *block, OpAsmSetValueNameFn setNameFn) const {
  auto *parentOp = block->getParentOp();
  auto component = dyn_cast<ComponentOp>(parentOp);
  // Currently only support named block arguments for components.
  if (component == nullptr)
    return;

  auto ports = component.portNames();
  for (size_t i = 0, e = block->getNumArguments(); i != e; ++i)
    setNameFn(block->getArgument(i), ports[i].cast<StringAttr>().getValue());
}

// Provide implementations for the enums and attributes we use.
#include "circt/Dialect/Calyx/CalyxAttributes.cpp.inc"
#include "circt/Dialect/Calyx/CalyxDialect.cpp.inc"
#include "circt/Dialect/Calyx/CalyxEnums.cpp.inc"

void CalyxDialect::initialize() {

  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/Calyx/Calyx.cpp.inc"
  >();

  // Register interface implementations.
  addInterfaces<CalyxOpAsmDialectInterface>();
}