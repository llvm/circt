//===- RangeAnalysisComb.cpp - Lower some ops in comb -------------------*- C++ -*-===//
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


struct CombAddNarrow : public mlir::OpRewritePattern<comb::AddOp> {
  CombAddNarrow(MLIRContext *context, DataFlowSolver &s) 
      : OpRewritePattern(context), solver(s) {}
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(comb::AddOp op,
                                PatternRewriter &rewriter) const override {
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
    else 
      return rewriter.notifyMatchFailure(op, "Could remove " + std::to_string(remove_width) + " bits - but didn't yet...");
  }

  private:
  DataFlowSolver &solver;
};

namespace {
class RangeAnalysisCombPass : public circt::comb::impl::RangeAnalysisCombBase<RangeAnalysisCombPass> {
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

void comb::populateCombNarrowingPatterns(
  RewritePatternSet &patterns, DataFlowSolver &solver) {
  patterns.add<CombAddNarrow>(patterns.getContext(), solver);
}