//===- LLHDCanonicalization.cpp - Register LLHD Canonicalization Patterns -===//
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "circt/Dialect/LLHD/IR/LLHDCanonicalization.inc"

struct DynExtractSliceWithConstantStart
    : public mlir::OpRewritePattern<llhd::DynExtractSliceOp> {
  DynExtractSliceWithConstantStart(mlir::MLIRContext *context)
      : OpRewritePattern<llhd::DynExtractSliceOp>(context, /*benefit=*/1) {}

  mlir::LogicalResult
  matchAndRewrite(llhd::DynExtractSliceOp op,
                  mlir::PatternRewriter &rewriter) const override {
    IntegerAttr intAttr;
    if (mlir::matchPattern(op.start(), m_Constant<IntegerAttr>(&intAttr))) {
      rewriter.replaceOpWithNewOp<llhd::ExtractSliceOp>(
          op, op.result().getType(), op.target(),
          rewriter.getIndexAttr(intAttr.getValue().getZExtValue()));
      return success();
    }
    return failure();
  }
};

struct DynExtractElementWithConstantIndex
    : public mlir::OpRewritePattern<llhd::DynExtractElementOp> {
  DynExtractElementWithConstantIndex(mlir::MLIRContext *context)
      : OpRewritePattern<llhd::DynExtractElementOp>(context,
                                                    /*benefit=*/1) {}

  mlir::LogicalResult
  matchAndRewrite(llhd::DynExtractElementOp op,
                  mlir::PatternRewriter &rewriter) const override {
    IntegerAttr intAttr;
    if (mlir::matchPattern(op.index(), m_Constant<IntegerAttr>(&intAttr))) {
      rewriter.replaceOpWithNewOp<llhd::ExtractElementOp>(
          op, op.result().getType(), op.target(),
          rewriter.getIndexAttr(intAttr.getValue().getZExtValue()));
      return success();
    }
    return failure();
  }
};
} // anonymous namespace

void llhd::XorOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<XorAllBitsSet>(context);
}

void llhd::NotOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<NotOfEq, NotOfNeq>(context);
}

void llhd::EqOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                             MLIRContext *context) {
  results.insert<BooleanEqToXor>(context);
}

void llhd::NeqOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<BooleanNeqToXor>(context);
}

void llhd::DynExtractSliceOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<DynExtractSliceWithConstantStart>(context);
}

void llhd::DynExtractElementOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<DynExtractElementWithConstantIndex>(context);
}
