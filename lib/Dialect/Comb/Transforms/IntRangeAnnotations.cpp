//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Comb/CombPasses.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

using namespace circt;
using namespace circt::comb;
using namespace mlir;
using namespace mlir::dataflow;

namespace circt {
namespace comb {
#define GEN_PASS_DEF_COMBOVERFLOWANNOTATING
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

namespace {
template <typename CombOpTy>
struct CombOpAnnotate : public OpRewritePattern<CombOpTy> {
  CombOpAnnotate(MLIRContext *context, DataFlowSolver &s)
      : OpRewritePattern<CombOpTy>(context), solver(s) {}

  LogicalResult matchAndRewrite(CombOpTy op,
                                PatternRewriter &rewriter) const override {
    // Use LLVM nuw attribute to avoid confusion with IEEE overflow flags
    if (op->hasAttr("comb.nuw"))
      return failure();

    if (op->getNumOperands() != 2 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(
          op, "Only support binary operations with one result");

    assert(isa<comb::AddOp>(op) || isa<comb::MulOp>(op) ||
           isa<comb::SubOp>(op));

    SmallVector<ConstantIntRanges> ranges;
    if (failed(collectRanges(solver, op->getOperands(), ranges)))
      return rewriter.notifyMatchFailure(op, "input without specified range");

    bool overflowed = false;
    // overflowed = a.max op b.max > 2^(bitwidth)-1
    auto a = ranges[0].umax();
    auto b = ranges[1].umax();

    // Only use the overflow flag - need to explicitly discard the result.
    // Cannot use operators intervals since they will be truncated to the
    // result's bitwidth.
    if (isa<comb::AddOp>(op))
      (void)a.uadd_ov(b, overflowed);

    if (isa<comb::MulOp>(op))
      (void)a.umul_ov(b, overflowed);

    if (!overflowed)
      op->setAttr("comb.nuw", UnitAttr::get(op->getContext()));

    return success();
  }

private:
  DataFlowSolver &solver;
};

struct CombOverflowAnnotatingPass
    : comb::impl::CombOverflowAnnotatingBase<CombOverflowAnnotatingPass> {

  using CombOverflowAnnotatingBase::CombOverflowAnnotatingBase;
  void runOnOperation() override;
};
} // namespace

void CombOverflowAnnotatingPass::runOnOperation() {
  Operation *op = getOperation();
  MLIRContext *ctx = op->getContext();
  DataFlowSolver solver;
  solver.load<DeadCodeAnalysis>();
  solver.load<IntegerRangeAnalysis>();
  if (failed(solver.initializeAndRun(op)))
    return signalPassFailure();

  RewritePatternSet patterns(ctx);
  // TODO: determine how to support subtraction
  patterns.add<CombOpAnnotate<comb::AddOp>, CombOpAnnotate<comb::MulOp>>(
      patterns.getContext(), solver);

  walkAndApplyPatterns(op, std::move(patterns));
}
