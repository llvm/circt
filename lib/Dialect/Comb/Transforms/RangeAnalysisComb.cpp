//===- RangeAnalysisComb.cpp - Lower some ops in comb -------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Comb/CombPasses.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace circt;
using namespace circt::comb;
using namespace mlir;
using namespace mlir::dataflow;

namespace circt {
namespace comb {
#define GEN_PASS_DEF_RANGEANALYSISCOMB
#include "circt/Dialect/Comb/Passes.h.inc"
} // namespace comb
} // namespace circt

/// Gather ranges for all the values in `values`. Appends to the existing
/// vector.
static LogicalResult collectRanges(DataFlowSolver &solver, ValueRange values,
                                   SmallVectorImpl<ConstantIntRanges> &ranges) {
  for (Value val : values) {
    auto *maybeInferredRange =
        solver.lookupState<IntegerValueRangeLattice>(val);
    if (!maybeInferredRange || maybeInferredRange->getValue().isUninitialized())
      return failure();

    const ConstantIntRanges &inferredRange =
        maybeInferredRange->getValue().getValue();
    ranges.push_back(inferredRange);
  }
  return success();
}

template <typename CombOpTy>
struct CombOpNarrow : public mlir::OpRewritePattern<CombOpTy> {
  CombOpNarrow(MLIRContext *context, DataFlowSolver &s)
      : OpRewritePattern<CombOpTy>(context), solver(s) {}

  LogicalResult matchAndRewrite(CombOpTy op,
                                PatternRewriter &rewriter) const override {

    auto op_width = op.getType().getIntOrFloatBitWidth();

    SmallVector<ConstantIntRanges> ranges;
    if (failed(collectRanges(solver, op->getOperands(), ranges)))
      return rewriter.notifyMatchFailure(op, "input without specified range");
    if (failed(collectRanges(solver, op->getResults(), ranges)))
      return rewriter.notifyMatchFailure(op, "output without specified range");

    auto remove_width = ranges[0].umax().countLeadingZeros();
    for (const ConstantIntRanges &range : ranges) {
      auto range_can_remove = range.umax().countLeadingZeros();
      remove_width = std::min(remove_width, range_can_remove);
    }
    if (remove_width == 0)
      return rewriter.notifyMatchFailure(op, "no bits to remove");
    else if (remove_width == op_width)
      return rewriter.notifyMatchFailure(
          op, "all bits to remove - replace by zero");
    else {
      Value lhs = op.getOperand(0);
      Value rhs = op.getOperand(1);

      Location loc = op.getLoc();
      auto newWidth = op_width - remove_width;
      // Create a replacement type for the extracted bits
      auto replaceType = rewriter.getIntegerType(newWidth);

      // Extract the lsbs from each operand
      auto extractLhsOp =
          rewriter.create<comb::ExtractOp>(loc, replaceType, lhs, 0);
      auto extractRhsOp =
          rewriter.create<comb::ExtractOp>(loc, replaceType, rhs, 0);
      auto narrowOp =
          rewriter.create<CombOpTy>(loc, extractLhsOp, extractRhsOp);

      auto zero =
          rewriter.create<hw::ConstantOp>(loc, APInt::getZero(remove_width));
      auto replaceOp = rewriter.create<comb::ConcatOp>(
          loc, op.getType(), ValueRange{zero, narrowOp});
      // Replace the original operation with the new one
      rewriter.replaceOp(op, replaceOp);
      return success();
    }
  }

private:
  DataFlowSolver &solver;
};

namespace {
class RangeAnalysisCombPass
    : public circt::comb::impl::RangeAnalysisCombBase<RangeAnalysisCombPass> {
public:
  using RangeAnalysisCombBase::RangeAnalysisCombBase;
  void runOnOperation() override;
};
} // namespace

void RangeAnalysisCombPass::runOnOperation() {
  Operation *op = getOperation();
  MLIRContext *ctx = op->getContext();
  DataFlowSolver solver;
  solver.load<DeadCodeAnalysis>();
  solver.load<IntegerRangeAnalysis>();
  if (failed(solver.initializeAndRun(op)))
    return signalPassFailure();

  RewritePatternSet patterns(ctx);
  populateCombNarrowingPatterns(patterns, solver);
  GreedyRewriteConfig config;
  // We specifically need bottom-up traversal as cmpi pattern needs range
  // data, attached to its original argument values.
  config.useTopDownTraversal = false;

  if (failed(applyPatternsGreedily(op, std::move(patterns), config)))
    signalPassFailure();
}

void comb::populateCombNarrowingPatterns(RewritePatternSet &patterns,
                                         DataFlowSolver &solver) {
  patterns.add<CombOpNarrow<comb::AddOp>, CombOpNarrow<comb::MulOp>,
               CombOpNarrow<comb::SubOp>>(patterns.getContext(), solver);
}