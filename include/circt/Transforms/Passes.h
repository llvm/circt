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

#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <limits>

namespace circt {

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL

enum class OpCountEmissionFormat {
  // Specify the format for op count data emission
  Readable,
  ReadableSorted,
  JSON
};

#include "circt/Transforms/Passes.h.inc"

void populateArithToCombPatterns(mlir::RewritePatternSet &patterns,
                                 TypeConverter &typeConverter);

std::unique_ptr<mlir::Pass>
createMapArithToCombPass(bool enableBestEffortLowering = false);
std::unique_ptr<mlir::Pass> createConvertIndexToUIntPass();
std::unique_ptr<mlir::Pass> createFlattenMemRefPass();
std::unique_ptr<mlir::Pass> createFlattenMemRefCallsPass();
std::unique_ptr<mlir::Pass> createStripDebugInfoWithPredPass(
    const std::function<bool(mlir::Location)> &pred);
std::unique_ptr<mlir::Pass> createMaximizeSSAPass();
std::unique_ptr<mlir::Pass> createInsertMergeBlocksPass();
std::unique_ptr<mlir::Pass> createPrintOpCountPass();
std::unique_ptr<mlir::Pass>
createMemoryBankingPass(ArrayRef<unsigned> bankingFactors = {},
                        ArrayRef<unsigned> bankingDimensions = {});
std::unique_ptr<mlir::Pass> createIndexSwitchToIfPass();
std::unique_ptr<mlir::Pass> createHierarchicalRunner(
    const std::string &topName,
    llvm::function_ref<void(mlir::OpPassManager &)> pipeline,
    bool includeBoundInstances = false);

//===----------------------------------------------------------------------===//
// Utility functions.
//===----------------------------------------------------------------------===//

// Returns true if the provided memref is considered unidimensional (having a
// shape of size 1).
bool isUniDimensional(mlir::MemRefType memref);

// Returns true if the region is into maximal SSA form i.e., if all the values
// within the region are in maximal SSA form.
bool isRegionSSAMaximized(Region &region);

/// Strategy class to control the behavior of SSA maximization. The class
/// exposes overridable filter functions to dynamically select which blocks,
/// block arguments, operations, and operation results should be put into
/// maximal SSA form. All filter functions should return true whenever the
/// entity they operate on should be considered for SSA maximization. By
/// default, all filter functions always return true.
class SSAMaximizationStrategy {
public:
  /// Determines whether a block should have the values it defines (i.e., block
  /// arguments and operation results within the block) SSA maximized.
  virtual bool maximizeBlock(Block *block);
  /// Determines whether a block argument should be SSA maximized.
  virtual bool maximizeArgument(BlockArgument arg);
  /// Determines whether an operation should have its results SSA maximized.
  virtual bool maximizeOp(Operation *op);
  /// Determines whether an operation's result should be SSA maximized.
  virtual bool maximizeResult(OpResult res);

  virtual ~SSAMaximizationStrategy() = default;
};

/// Converts a single value within a function into maximal SSA form. This
/// removes any implicit dataflow of this specific value within the enclosing
/// function. The function adds new block arguments wherever necessary to carry
/// the value explicitly between blocks.
/// Succeeds when it was possible to convert the value into maximal SSA form.
LogicalResult maximizeSSA(Value value, PatternRewriter &rewriter);

/// Considers all of an operation's results for SSA maximization, following a
/// provided strategy. This removes any implicit dataflow of the selected
/// operation's results within the enclosing function. The function adds new
/// block arguments wherever necessary to carry the results explicitly between
/// blocks. Succeeds when it was possible to convert the selected operation's
/// results into maximal SSA form.
LogicalResult maximizeSSA(Operation *op, SSAMaximizationStrategy &strategy,
                          PatternRewriter &rewriter);

/// Considers all values defined by a block (i.e., block arguments and operation
/// results within the block) for SSA maximization, following a provided
/// strategy. This removes any implicit dataflow of the selected values within
/// the enclosing function. The function adds new block arguments wherever
/// necessary to carry the values explicitly between blocks. Succeeds when it
/// was possible to convert the selected values defined by the block into
/// maximal SSA form.
LogicalResult maximizeSSA(Block *block, SSAMaximizationStrategy &strategy,
                          PatternRewriter &rewriter);

/// Considers all blocks within a region for SSA maximization, following a
/// provided strategy. This removes any implicit dataflow of the values defined
/// by selected blocks within the region. The function adds new block arguments
/// wherever necessary to carry the region's values explicitly between blocks.
/// Succeeds when it was possible to convert all of the values defined by
/// selected blocks into maximal SSA form.
LogicalResult maximizeSSA(Region &region, SSAMaximizationStrategy &strategy,
                          PatternRewriter &rewriter);

/// Manually run merge block insertion on a region.
///
/// This transformation does treat loops like a single block and thus does not
/// affect them.
LogicalResult insertMergeBlocks(mlir::Region &r,
                                mlir::ConversionPatternRewriter &rewriter);

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt/Transforms/Passes.h.inc"

} // namespace circt

#endif // CIRCT_TRANSFORMS_PASSES_H
