//===- SimUtils.h - Sim utility entry points --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines utility functions for the Sim dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SIM_SIMUTILS_H
#define CIRCT_DIALECT_SIM_SIMUTILS_H

#include "circt/Dialect/Sim/SimOps.h"

namespace circt {
namespace sim {

/// Erase a print op and cascade-delete dead, side-effect-free producer ops in
/// its reachable dependency graph.
///
/// Precondition: the reachable deletable producer graph is a DAG.
/// In a DAG, this utility fully removes all deletable ops that become dead.
///
/// If the graph is not a DAG, the behavior is still defined: any strongly
/// connected component in that graph cannot be fully cleaned up by this local
/// dead-use criterion and may remain after erasing the root print.
///
/// Note: cyclic `sim.fmt.concat` dependencies are illegal IR by Sim dialect
/// invariants; callers should run this helper on verifier-clean IR.
void cascadeErasePrint(PrintFormattedOp op, mlir::RewriterBase &rewriter);
void cascadeErasePrint(PrintFormattedProcOp op, mlir::RewriterBase &rewriter);

/// TODO: Add explicit cycle detection helper if callers need local validation.

} // namespace sim
} // namespace circt

#endif // CIRCT_DIALECT_SIM_SIMUTILS_H
