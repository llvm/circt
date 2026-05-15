//===- SimTransforms.h - Sim transform helpers -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares reusable transformation helpers for the Sim dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SIM_SIMTRANSFORMS_H
#define CIRCT_DIALECT_SIM_SIMTRANSFORMS_H

#include "mlir/IR/Builders.h"
#include "llvm/ADT/ArrayRef.h"

namespace circt {
namespace sim {

struct PrintProceduralizationRequest {
  mlir::Location loc;
  mlir::Value input;
  mlir::Value condition;
  mlir::Value stream;

  /// Operation used for diagnostics, if any.
  mlir::Operation *anchorOp = nullptr;

  /// Operation to erase once the request has been proceduralized, if any.
  mlir::Operation *cleanupRoot = nullptr;
};

/// Lower a list of same-clock print requests into a shared `hw.triggered`
/// region containing `sim.proc.print` operations.
mlir::LogicalResult proceduralizePrintsForClock(
    mlir::OpBuilder &builder, mlir::Value clock,
    llvm::ArrayRef<PrintProceduralizationRequest> printRequests);

} // namespace sim
} // namespace circt

#endif // CIRCT_DIALECT_SIM_SIMTRANSFORMS_H
