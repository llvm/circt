//===- FIRRTLOps.h - Declare FIRRTL dialect operations ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation class for the FIRRTL IR.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_OPS_H
#define CIRCT_DIALECT_FIRRTL_OPS_H

#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace circt {
namespace firrtl {

/// Return true if the specified operation is a firrtl expression.
bool isExpression(Operation *op);

/// This holds the name and type that describes the module's ports.
struct ModulePortInfo {
  StringAttr name;
  FIRRTLType type;

  StringRef getName() const { return name ? name.getValue() : ""; }

  /// Return true if this is a simple output-only port.
  bool isOutput() { return type.isa<FlipType>(); }

  /// Return true if this is a simple input-only port.
  bool isInput() {
    auto flags = type.getRecursiveTypeProperties();
    // isPassive & !containsAnalog.
    return flags.first && !flags.second;
  }

  /// Return true if this is an inout port.
  bool isInOut() { return !isOutput() && !isInput(); }
};

/// Return the function type that corresponds to a module.
FunctionType getModuleType(Operation *op);

/// This function can extract information about ports from a module and an
/// extmodule.
SmallVector<ModulePortInfo> getModulePortInfo(Operation *op);

/// Given an FModule or ExtModule, return the name of the specified port number.
StringAttr getModulePortName(Operation *op, size_t portIndex);

/// Returns true if the type is a bundle or a flip of a bundle.
bool isBundleType(Type type);

/// Returns true if the value results from an expression with duplex flow.
/// Duplex values have special treatment in bundle connect operations, and their
/// flip orientation is not used to determine the direction of each pairwise
/// connect.
bool isDuplexValue(Value val);

} // namespace firrtl
} // namespace circt

#define GET_OP_CLASSES
#include "circt/Dialect/FIRRTL/FIRRTL.h.inc"

#endif // CIRCT_DIALECT_FIRRTL_OPS_H
