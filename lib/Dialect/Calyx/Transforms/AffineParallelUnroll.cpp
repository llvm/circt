//===- AffineParallelUnroll.cpp - Unroll AffineParallelOp ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unroll AffineParallelOp to facilitate lowering to Calyx ParOp.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Calyx/CalyxPasses.h"
#include "circt/Support/LLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace circt {
namespace calyx {
#define GEN_PASS_DEF_AFFINEPARALLELUNROLL
#include "circt/Dialect/Calyx/CalyxPasses.h.inc"
} // namespace calyx
} // namespace circt

using namespace circt;
using namespace mlir;
using namespace mlir::affine;
using namespace mlir::arith;

namespace {

struct AffineParallelUnroll : public OpRewritePattern<AffineParallelOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineParallelOp affineParallelOp,
                                PatternRewriter &rewriter) const override {
    if (affineParallelOp->hasAttr("calyx.parallel"))
      // We assume that having "calyx.parallel" attribute means that it has
      // already been unrolled.
      return failure();

    if (!affineParallelOp.getResults().empty()) {
      affineParallelOp.emitError(
          "affine.parallel with reductions is not supported yet");
      return failure();
    }

    Location loc = affineParallelOp.getLoc();

    rewriter.setInsertionPointAfter(affineParallelOp);
    // Create a single-iteration parallel loop op and mark its special by
    // setting the "calyx.parallel" attribute.
    AffineMap lbMap = AffineMap::get(0, 0, rewriter.getAffineConstantExpr(0),
                                     rewriter.getContext());
    AffineMap ubMap = AffineMap::get(0, 0, rewriter.getAffineConstantExpr(1),
                                     rewriter.getContext());
    auto newParallelOp = rewriter.create<AffineParallelOp>(
        loc, /*resultTypes=*/TypeRange(),
        /*reductions=*/SmallVector<arith::AtomicRMWKind>(),
        /*lowerBoundsMap=*/lbMap, /*lowerBoundsOperands=*/SmallVector<Value>(),
        /*upperBoundsMap=*/ubMap, /*upperBoundsOperands=*/SmallVector<Value>(),
        /*steps=*/SmallVector<int64_t>({1}));
    newParallelOp->setAttr("calyx.parallel", rewriter.getBoolAttr(true));

    SmallVector<int64_t> pLoopLowerBounds =
        affineParallelOp.getLowerBoundsMap().getConstantResults();
    if (pLoopLowerBounds.empty()) {
      affineParallelOp.emitError(
          "affine.parallel must have constant lower bounds");
      return failure();
    }
    SmallVector<int64_t> pLoopUpperBounds =
        affineParallelOp.getUpperBoundsMap().getConstantResults();
    if (pLoopUpperBounds.empty()) {
      affineParallelOp.emitError(
          "affine.parallel must have constant upper bounds");
      return failure();
    }
    SmallVector<int64_t, 8> pLoopSteps = affineParallelOp.getSteps();

    Block *pLoopBody = affineParallelOp.getBody();
    MutableArrayRef<BlockArgument> pLoopIVs = affineParallelOp.getIVs();

    OpBuilder insideBuilder(newParallelOp);
    SmallVector<int64_t> indices = pLoopLowerBounds;
    while (true) {
      insideBuilder.setInsertionPointToStart(newParallelOp.getBody());
      // Create an `scf.execute_region` to wrap each unrolled block since
      // `affine.parallel` requires only one block in the body region.
      auto executeRegionOp =
          insideBuilder.create<scf::ExecuteRegionOp>(loc, TypeRange{});
      Region &executeRegionRegion = executeRegionOp.getRegion();
      Block *executeRegionBlock = &executeRegionRegion.emplaceBlock();

      OpBuilder regionBuilder(executeRegionOp);
      // Each iteration starts with a fresh mapping, so each new block’s
      // argument of a region-based operation (such as `affine.for`) get
      // re-mapped independently.
      IRMapping operandMap;
      regionBuilder.setInsertionPointToEnd(executeRegionBlock);
      // Map induction variables to constant indices
      for (unsigned i = 0; i < indices.size(); ++i) {
        Value ivConstant =
            regionBuilder.create<arith::ConstantIndexOp>(loc, indices[i]);
        operandMap.map(pLoopIVs[i], ivConstant);
      }

      for (auto it = pLoopBody->begin(); it != std::prev(pLoopBody->end());
           ++it)
        regionBuilder.clone(*it, operandMap);

      // A terminator should always be inserted in `scf.execute_region`'s block.
      regionBuilder.create<scf::YieldOp>(loc);

      // Increment indices using `step`.
      bool done = false;
      for (int dim = indices.size() - 1; dim >= 0; --dim) {
        indices[dim] += pLoopSteps[dim];
        if (indices[dim] < pLoopUpperBounds[dim])
          break;
        indices[dim] = pLoopLowerBounds[dim];
        if (dim == 0)
          // All combinations have been generated
          done = true;
      }
      if (done)
        break;
    }

    rewriter.replaceOp(affineParallelOp, newParallelOp);

    return success();
  }
};

struct AffineParallelUnrollPass
    : public circt::calyx::impl::AffineParallelUnrollBase<
          AffineParallelUnrollPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::scf::SCFDialect>();
  }
  void runOnOperation() override;
};

} // end anonymous namespace

void AffineParallelUnrollPass::runOnOperation() {
  auto *ctx = &getContext();
  ConversionTarget target(*ctx);

  RewritePatternSet patterns(ctx);
  patterns.add<AffineParallelUnroll>(ctx);

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> circt::calyx::createAffineParallelUnrollPass() {
  return std::make_unique<AffineParallelUnrollPass>();
}
