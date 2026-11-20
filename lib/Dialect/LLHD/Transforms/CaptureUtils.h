//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helper to capture values across `llhd.wait` ops.
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef CIRCT_DIALECT_LLHD_TRANSFORMS_CAPTUREUTILS_H
#define CIRCT_DIALECT_LLHD_TRANSFORMS_CAPTUREUTILS_H

#include "circt/Dialect/LLHD/LLHDOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/IR/Dominance.h"

namespace circt {
namespace llhd {

/// Add a value as block argument to a list of wait ops and update uses of the
/// value to use the added block arguments as appropriate. This may insert
/// additional block arguments in case the value and added block arguments both
/// reach the same block.
void captureAcrossWait(Value value, ArrayRef<WaitOp> waitOps,
                       mlir::Liveness &liveness,
                       mlir::DominanceInfo &dominance);

} // namespace llhd
} // namespace circt

#endif // CIRCT_DIALECT_LLHD_TRANSFORMS_CAPTUREUTILS_H
