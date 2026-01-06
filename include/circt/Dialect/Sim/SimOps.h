//===- SimOps.h - Declare Sim dialect operations ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the Sim dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SIM_SIMOPS_H
#define CIRCT_DIALECT_SIM_SIMOPS_H

#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "circt/Dialect/Sim/SimDialect.h"
#include "circt/Dialect/Sim/SimTypes.h"
#include "circt/Support/BuilderUtils.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

#define GET_OP_CLASSES
#include "circt/Dialect/Sim/Sim.h.inc"

namespace circt {
namespace sim {

/// Returns the value operand of a value formatting operation.
/// Returns a null value for all other operations.
static inline mlir::Value getFormattedValue(mlir::Operation *fmtOp) {
  if (auto fmt = llvm::dyn_cast_or_null<circt::sim::FormatBinOp>(fmtOp))
    return fmt.getValue();
  if (auto fmt = llvm::dyn_cast_or_null<circt::sim::FormatDecOp>(fmtOp))
    return fmt.getValue();
  if (auto fmt = llvm::dyn_cast_or_null<circt::sim::FormatOctOp>(fmtOp))
    return fmt.getValue();
  if (auto fmt = llvm::dyn_cast_or_null<circt::sim::FormatHexOp>(fmtOp))
    return fmt.getValue();
  if (auto fmt = llvm::dyn_cast_or_null<circt::sim::FormatCharOp>(fmtOp))
    return fmt.getValue();
  return {};
}

} // namespace sim
} // namespace circt

#endif // CIRCT_DIALECT_SIM_SIMOPS_H
