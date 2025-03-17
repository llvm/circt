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
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "calyx-affine-to-scf"

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
      steps.push_back(rewriter.create<arith::ConstantIndexOp>(loc, step));

    SmallVector<Value, 8> upperBoundTuple;
    SmallVector<Value, 8> lowerBoundTuple;
    lowerBoundTuple.reserve(affineParallelOp.getNumDims());
    upperBoundTuple.reserve(affineParallelOp.getNumDims());
    // Emit IR computing the lower and upper bound by expanding the map
    // expression.
    for (unsigned i = 0, e = affineParallelOp.getNumDims(); i < e; ++i) {
      Value lower =
          lowerAffineMapMax(rewriter, loc, affineParallelOp.getLowerBoundMap(i),
                            affineParallelOp.getLowerBoundsOperands());
      if (!lower)
        return rewriter.notifyMatchFailure(affineParallelOp,
                                           "couldn't convert lower bounds");
      lowerBoundTuple.push_back(lower);

      Value upper =
          lowerAffineMapMin(rewriter, loc, affineParallelOp.getUpperBoundMap(i),
                            affineParallelOp.getUpperBoundsOperands());
      if (!upper)
        return rewriter.notifyMatchFailure(affineParallelOp,
                                           "couldn't convert upper bounds");
      upperBoundTuple.push_back(upper);
    }

    auto affineParallelTerminator = cast<affine::AffineYieldOp>(
        affineParallelOp.getBody()->getTerminator());

    scf::ParallelOp scfParallelOp = rewriter.create<scf::ParallelOp>(
        loc, lowerBoundTuple, upperBoundTuple, steps,
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

private:
  /// Given a range of values, emit the code that reduces them with "min" or
  /// "max" depending on the provided comparison predicate, sgt for max and slt
  /// for min.
  ///
  /// Multiple values are scanned in a linear sequence.  This creates a data
  /// dependences that wouldn't exist in a tree reduction, but is easier to
  /// recognize as a reduction by the subsequent passes.
  Value buildMinMaxReductionSeq(Location loc, arith::CmpIPredicate predicate,
                                ValueRange values, OpBuilder &builder) const {
    assert(!values.empty() && "empty min/max chain");
    assert(predicate == arith::CmpIPredicate::sgt ||
           predicate == arith::CmpIPredicate::slt);

    auto valueIt = values.begin();
    Value value = *valueIt++;
    for (; valueIt != values.end(); ++valueIt) {
      if (predicate == arith::CmpIPredicate::sgt)
        value = builder.create<arith::MaxSIOp>(loc, value, *valueIt);
      else
        value = builder.create<arith::MinSIOp>(loc, value, *valueIt);
    }

    return value;
  }

  /// Emit instructions that correspond to computing the minimum value among the
  /// values of a (potentially) multi-output affine map applied to `operands`.
  Value lowerAffineMapMin(OpBuilder &builder, Location loc, AffineMap map,
                          ValueRange operands) const {
    if (auto values =
            mlir::affine::expandAffineMap(builder, loc, map, operands))
      return buildMinMaxReductionSeq(loc, arith::CmpIPredicate::slt, *values,
                                     builder);
    return nullptr;
  }

  /// Emit instructions that correspond to computing the maximum value among the
  /// values of a (potentially) multi-output affine map applied to `operands`.
  Value lowerAffineMapMax(OpBuilder &builder, Location loc, AffineMap map,
                          ValueRange operands) const {
    if (auto values =
            mlir::affine::expandAffineMap(builder, loc, map, operands))
      return buildMinMaxReductionSeq(loc, arith::CmpIPredicate::sgt, *values,
                                     builder);
    return nullptr;
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
