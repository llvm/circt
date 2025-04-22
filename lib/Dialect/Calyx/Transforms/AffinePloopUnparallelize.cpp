//===- AffinePloopUnparallize.cpp
//----------------------------------------------------===//
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
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace circt {
namespace calyx {
#define GEN_PASS_DEF_AFFINEPLOOPUNPARALLELIZE
#include "circt/Dialect/Calyx/CalyxPasses.h.inc"
} // namespace calyx
} // namespace circt

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::memref;
using namespace mlir::scf;
using namespace mlir::func;
using namespace circt;

class AffinePloopUnparallelize
    : public OpConversionPattern<affine::AffineParallelOp> {
  using OpConversionPattern::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(affine::AffineParallelOp affineParallelOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (affineParallelOp.getIVs().size() != 1)
      return rewriter.notifyMatchFailure(affineParallelOp,
                                         "currently only support single IV");

    auto loc = affineParallelOp.getLoc();
    auto upperBoundTuple = mlir::affine::expandAffineMap(
        rewriter, loc, affineParallelOp.getUpperBoundsMap(),
        affineParallelOp.getUpperBoundsOperands());
    if (!upperBoundTuple)
      return rewriter.notifyMatchFailure(affineParallelOp,
                                         "does not have upper bounds");
    Value upperBound = (*upperBoundTuple)[0];

    auto lowerBoundTuple = mlir::affine::expandAffineMap(
        rewriter, loc, affineParallelOp.getLowerBoundsMap(),
        affineParallelOp.getLowerBoundsOperands());
    if (!lowerBoundTuple)
      return rewriter.notifyMatchFailure(affineParallelOp,
                                         "does not have lower bounds");
    Value lowerBound = (*lowerBoundTuple)[0];

    auto step = affineParallelOp.getSteps()[0];

    auto factorAttr =
        affineParallelOp->getAttrOfType<IntegerAttr>("unparallelize.factor");
    if (!factorAttr)
      return rewriter.notifyMatchFailure(affineParallelOp,
                                         "Missing 'unparallelize.factor'");

    int64_t factor = factorAttr.getInt();

    SmallVector<scf::IndexSwitchOp> simplifiableIndexSwitchOps =
        collectSimplifiableIndexSwitchOps(affineParallelOp, factor);

    auto outerLoop = rewriter.create<affine::AffineForOp>(
        loc, lowerBound, rewriter.getDimIdentityMap(), upperBound,
        rewriter.getDimIdentityMap(), step * factor);

    rewriter.setInsertionPointToStart(outerLoop.getBody());
    AffineMap lbMap = AffineMap::get(
        /*dimCount=*/0, /*symbolCount=*/0,
        /*results=*/rewriter.getAffineConstantExpr(0), rewriter.getContext());
    AffineMap ubMap = AffineMap::get(
        0, 0, rewriter.getAffineConstantExpr(factor), rewriter.getContext());
    auto innerParallel = rewriter.create<affine::AffineParallelOp>(
        loc, /*resultTypes=*/TypeRange(),
        /*reductions=*/SmallVector<arith::AtomicRMWKind>(),
        /*lowerBoundsMap=*/lbMap, /*lowerBoundsOperands=*/SmallVector<Value>(),
        /*upperBoundsMap=*/ubMap, /*upperBoundsOperands=*/SmallVector<Value>(),
        /*steps=*/SmallVector<int64_t>({step}));

    if (!innerParallel.getBody()->empty()) {
      Operation &lastOp = innerParallel.getBody()->back();
      if (isa<affine::AffineYieldOp>(lastOp))
        lastOp.erase();
    }
    rewriter.setInsertionPointToStart(innerParallel.getBody());

    // `newIndex` will be the newly created `affine.for`'s IV added with the
    // inner `affine.parallel`'s IV.
    auto addMap = AffineMap::get(
        2, 0, rewriter.getAffineDimExpr(0) + rewriter.getAffineDimExpr(1),
        rewriter.getContext());

    auto newIndex = rewriter.create<affine::AffineApplyOp>(
        loc, addMap,
        ValueRange{outerLoop.getInductionVar(), innerParallel.getIVs()[0]});

    Block *srcBlock = affineParallelOp.getBody();
    Block *destBlock = innerParallel.getBody();

    // Move all operations except the terminator from `srcBlock` to `destBlock`.
    destBlock->getOperations().splice(
        destBlock->end(),          // insert at the end of `destBlock`
        srcBlock->getOperations(), // move ops from `srcBlock`
        srcBlock->begin(),         // start at beginning of `srcBlock`
        std::prev(srcBlock->end()) // stop before the terminator op
    );

    // Remap occurrences of the old induction variable in the moved ops.
    destBlock->walk([&](Operation *op) {
      for (OpOperand &operand : op->getOpOperands()) {
        if (operand.get() == affineParallelOp.getIVs()[0])
          operand.set(newIndex);
      }
    });

    rewriter.setInsertionPointToEnd(destBlock);
    rewriter.create<affine::AffineYieldOp>(loc);

    for (auto indexSwitchOp : simplifiableIndexSwitchOps) {
      indexSwitchOp.setOperand(innerParallel.getIVs().front());
    }

    return success();
  }

private:
  // Collect all simplifiable `scf.index_switch` ops in `affineParallelOp`. An
  // `scf.index_switch` op is simpliiable if its argument only depends on
  // `affineParallelOp`'s loop IV and if it's a result of a modulo expression.
  SmallVector<scf::IndexSwitchOp>
  collectSimplifiableIndexSwitchOps(affine::AffineParallelOp affineParallelOp,
                                    int64_t factor) const {
    SmallVector<scf::IndexSwitchOp> result;
    affineParallelOp->walk([&](scf::IndexSwitchOp indexSwitchOp) {
      auto switchArg = indexSwitchOp.getArg();
      auto affineApplyOp =
          dyn_cast_or_null<affine::AffineApplyOp>(switchArg.getDefiningOp());
      if (!affineApplyOp || affineApplyOp->getNumOperands() != 1 ||
          affineApplyOp->getNumResults() != 1)
        return WalkResult::advance();

      auto affineMap = affineApplyOp.getAffineMap();
      auto binExpr = dyn_cast<AffineBinaryOpExpr>(affineMap.getResult(0));
      if (!binExpr || binExpr.getKind() != AffineExprKind::Mod)
        return WalkResult::advance();

      if (affineApplyOp.getOperand(0) != affineParallelOp.getIVs().front())
        return WalkResult::advance();

      auto rhs = binExpr.getRHS();
      auto constRhs = dyn_cast<AffineConstantExpr>(rhs);
      if (!constRhs || factor != constRhs.getValue())
        return WalkResult::advance();

      result.push_back(indexSwitchOp);
      return WalkResult::advance();
    });
    return result;
  }
};

namespace {
class AffinePloopUnparallelizePass
    : public circt::calyx::impl::AffinePloopUnparallelizeBase<
          AffinePloopUnparallelizePass> {
  void runOnOperation() override;
};
} // namespace

void AffinePloopUnparallelizePass::runOnOperation() {
  MLIRContext *ctx = &getContext();

  ConversionTarget target(*ctx);
  target.addLegalDialect<arith::ArithDialect, memref::MemRefDialect,
                         scf::SCFDialect, affine::AffineDialect>();

  RewritePatternSet patterns(ctx);
  patterns.add<AffinePloopUnparallelize>(ctx);
  GreedyRewriteConfig config;
  config.strictMode = GreedyRewriteStrictness::ExistingOps;
  if (failed(
          applyPatternsGreedily(getOperation(), std::move(patterns), config))) {
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> circt::calyx::createAffinePloopUnparallelizePass() {
  return std::make_unique<AffinePloopUnparallelizePass>();
}
