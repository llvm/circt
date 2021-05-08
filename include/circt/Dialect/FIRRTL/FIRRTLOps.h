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

/// Return the portNames attribute for the specified module, which contains the
/// name for each port.
ArrayAttr getModulePortNames(Operation *module);

/// Given an FModule or ExtModule, return the name of the specified port number.
StringAttr getModulePortName(Operation *op, size_t portIndex);

/// Returns true if the type is a bundle or a flip of a bundle.
bool isBundleType(Type type);

/// Returns true if the value results from an expression with duplex flow.
/// Duplex values have special treatment in bundle connect operations, and their
/// flip orientation is not used to determine the direction of each pairwise
/// connect.
bool isDuplexValue(Value val);

enum class Flow { Source, Sink, Duplex };

/// Get a flow's reverse.
Flow swapFlow(Flow flow);

/// Compute the flow for a Value, \p val, as determined by the FIRRTL
/// specification.  This recursively walks backwards from \p val to the
/// declaration.  The resulting flow is a combination of the declaration flow
/// (output ports and instance inputs are sinks, registers and wires are duplex,
/// anything else is a source) and the number of intermediary flips.  An even
/// number of flips will result in the same flow as the declaration.  An odd
/// number of flips will result in reversed flow being returned.  The reverse of
/// source is sink.  The reverse of sink is source.  The reverse of duplex is
/// duplex.  The \p accumulatedFlow parameter sets the initial flow.  A user
/// should normally \a not have to change this from its default of \p
/// Flow::Source.
Flow foldFlow(Value val, Flow accumulatedFlow = Flow::Source);

enum class DeclKind { Port, Instance, Other };

DeclKind getDeclarationKind(Value val);

} // namespace firrtl
} // namespace circt

#define GET_OP_CLASSES
#include "circt/Dialect/FIRRTL/FIRRTL.h.inc"

#endif // CIRCT_DIALECT_FIRRTL_OPS_H
