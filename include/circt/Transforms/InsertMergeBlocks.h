//===- InsertMergeBlocks.h - Insert Merge Blocks ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TRANSFORMS_INSERTMERGEBLOCKS_H
#define CIRCT_TRANSFORMS_INSERTMERGEBLOCKS_H

#include "mlir/Transforms/DialectConversion.h"

namespace circt {

/// Insert additional blocks that serve as counterparts to the blocks that
/// diverged the control flow.
/// The resulting merge block tree is guaranteed to be a binary tree.
///
/// This transformation does not affect any blocks that are part of a loop.
mlir::LogicalResult
insertExplicitMergeBlocks(mlir::Region &r,
                          mlir::ConversionPatternRewriter &rewriter);

} // namespace circt

#endif // CIRCT_TRANSFORMS_INSERTMERGEBLOCKS_H
