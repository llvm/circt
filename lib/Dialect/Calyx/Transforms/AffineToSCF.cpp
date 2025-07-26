//===- AffineToSCF.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Calyx/CalyxPasses.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace circt {
namespace calyx {
#define GEN_PASS_DEF_AFFINETOSCF
#include "circt/Dialect/Calyx/CalyxPasses.h.inc"
} // namespace calyx
} // namespace circt

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::memref;
using namespace mlir::scf;
using namespace mlir::func;
using namespace circt;

class AffineParallelOpLowering
    : public OpConversionPattern<affine::AffineParallelOp> {
  using OpConversionPattern::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(affine::AffineParallelOp affineParallelOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto affineParallelSteps = affineParallelOp.getSteps();
    if (std::any_of(affineParallelSteps.begin(), affineParallelSteps.end(),
                    [](int step) { return step > 1; }) ||
        !affineParallelOp->getAttr("calyx.unroll"))
      return rewriter.notifyMatchFailure(
          affineParallelOp,
          "Please run the MLIR canonical '-lower-affine' pass.");

    if (!affineParallelOp.getResults().empty())
      return rewriter.notifyMatchFailure(
          affineParallelOp, "Currently doesn't support parallel reduction.");

    Location loc = affineParallelOp.getLoc();
    SmallVector<Value, 8> steps;
    for (int64_t step : affineParallelSteps)
      steps.push_back(arith::ConstantIndexOp::create(rewriter, loc, step));

    auto upperBoundTuple = mlir::affine::expandAffineMap(
        rewriter, loc, affineParallelOp.getUpperBoundsMap(),
        affineParallelOp.getUpperBoundsOperands());

    auto lowerBoundTuple = mlir::affine::expandAffineMap(
        rewriter, loc, affineParallelOp.getLowerBoundsMap(),
        affineParallelOp.getLowerBoundsOperands());

    auto affineParallelTerminator = cast<affine::AffineYieldOp>(
        affineParallelOp.getBody()->getTerminator());

    scf::ParallelOp scfParallelOp = scf::ParallelOp::create(
        rewriter, loc, *lowerBoundTuple, *upperBoundTuple, steps,
        /*bodyBuilderFn=*/nullptr);
    scfParallelOp->setAttr("calyx.unroll",
                           affineParallelOp->getAttr("calyx.unroll"));
    rewriter.eraseBlock(scfParallelOp.getBody());
    rewriter.inlineRegionBefore(affineParallelOp.getRegion(),
                                scfParallelOp.getRegion(),
                                scfParallelOp.getRegion().end());
    rewriter.replaceOp(affineParallelOp, scfParallelOp);
    rewriter.setInsertionPoint(affineParallelTerminator);
    rewriter.replaceOpWithNewOp<scf::ReduceOp>(affineParallelTerminator);

    return success();
  }
};

namespace {
class AffineToSCFPass
    : public circt::calyx::impl::AffineToSCFBase<AffineToSCFPass> {
  void runOnOperation() override;
};
} // namespace

void AffineToSCFPass::runOnOperation() {
  MLIRContext *ctx = &getContext();

  ConversionTarget target(*ctx);
  target.addLegalDialect<arith::ArithDialect, memref::MemRefDialect,
                         scf::SCFDialect>();

  RewritePatternSet patterns(ctx);
  patterns.add<AffineParallelOpLowering>(ctx);
  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> circt::calyx::createAffineToSCFPass() {
  return std::make_unique<AffineToSCFPass>();
}
