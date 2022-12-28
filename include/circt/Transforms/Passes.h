//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for CIRCT transformation passes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TRANSFORMS_PASSES_H
#define CIRCT_TRANSFORMS_PASSES_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <limits>

namespace circt {

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass> createFlattenMemRefPass();
std::unique_ptr<mlir::Pass> createFlattenMemRefCallsPass();
std::unique_ptr<mlir::Pass> createStripDebugInfoWithPredPass(
    const std::function<bool(mlir::Location)> &pred);
std::unique_ptr<mlir::Pass> createMaximizeSSAPass();

//===----------------------------------------------------------------------===//
// Utility functions.
//===----------------------------------------------------------------------===//

// Returns true if the provided memref is considered unidimensional (having a
// shape of size 1).
bool isUniDimensional(mlir::MemRefType memref);

// Returns true if the region is into maximal SSA form i.e., if all the values
// within the region are in maximal SSA form.
bool isRegionSSAMaximized(Region &region);

/// Converts a single value within a function into maximal SSA form. This
/// removes any implicit dataflow of this specific value within the enclosing
/// function. The function adds new block arguments wherever necessary to carry
/// the value explicitly between blocks.
/// Succeeds when it was possible to convert the value into maximal SSA form.
LogicalResult maximizeSSA(Value value, PatternRewriter &rewriter);

/// Converts an operation's results within a function into maximal SSA form.
/// This removes any implicit dataflow of the operation's results within the
/// enclosing function. The function adds new block arguments wherever necessary
/// to carry the results explicitly between blocks.
/// Succeeds when it was possible to convert the operation's results into
/// maximal SSA form.
LogicalResult maximizeSSA(Operation *op, PatternRewriter &rewriter);

/// Converts all values defined by a block (i.e., block arguments and block's
/// operations' results) within a function into maximal SSA form. This removes
/// any implicit dataflow of those values within the enclosing function. The
/// function adds new block arguments wherever necessary to carry the values
/// explicitly between blocks.
/// Succeeds when it was possible to convert the values defined by the block
/// into maximal SSA form.
LogicalResult maximizeSSA(Block *block, PatternRewriter &rewriter);

/// Converts a region within a function into maximal SSA form. This removes any
/// implicit dataflow of the region's values. The function adds new block
/// arguments wherever necessary to carry the region's values explicitly between
/// blocks.
//' Succeeds when it was possible to convert the all of the region's values into
// maximal SSA form.
LogicalResult maximizeSSA(Region &region, PatternRewriter &rewriter);

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt/Transforms/Passes.h.inc"

} // namespace circt

#endif // CIRCT_TRANSFORMS_PASSES_H
