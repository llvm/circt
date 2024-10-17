//===- CombToAIG.cpp - Comb to AIG Conversion Pass ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/CombToAIG.h"
#include "circt/Dialect/AIG/AIGOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace circt {
#define GEN_PASS_DEF_CONVERTCOMBTOAIG
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace circt;
using namespace comb;

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {

static Value lowerFullyAssociativeOp(Operation *op, OperandRange operands,
                                     ConversionPatternRewriter &rewriter) {
  Value lhs, rhs;
  switch (operands.size()) {
  case 0:
    assert(0 && "cannot be called with empty operand range");
    break;
  case 1:
    return operands[0];
  case 2:
    lhs = operands[0];
    rhs = operands[1];
    break;
  default:
    auto firstHalf = operands.size() / 2;
    lhs = lowerFullyAssociativeOp(op, operands.take_front(firstHalf), rewriter);
    rhs = lowerFullyAssociativeOp(op, operands.drop_front(firstHalf), rewriter);
    break;
  }

  OperationState state(op->getLoc(), op->getName());
  state.addOperands(ValueRange{lhs, rhs});
  state.addTypes(op->getResult(0).getType());
  auto *newOp = Operation::create(state);
  rewriter.insert(newOp);
  return newOp->getResult(0);
}

template <typename OpTy>
struct VariadicOpConversion : OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpTy::Adaptor;
  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto result = lowerFullyAssociativeOp(op, op.getOperands(), rewriter);
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Lower a comb::AndOp operation to aig::AndOp
struct CombAndOpConversion : OpConversionPattern<AndOp> {
  using OpConversionPattern<AndOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AndOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<aig::AndOp>(op, adaptor.getInputs());
    return success();
  }
};

/// Lower a comb::OrOp operation to aig::AndOp with invert flags
struct CombOrOpConversion : OpConversionPattern<OrOp> {
  using OpConversionPattern<OrOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Implement Or using And and invert flags: a | b = ~(~a & ~b)
    auto andOp = rewriter.create<aig::AndOp>(
        op.getLoc(), adaptor.getInputs()[0], adaptor.getInputs()[1],
        /*lhs_invert=*/true,
        /*rhs_invert=*/true);
    rewriter.replaceOpWithNewOp<aig::AndOp>(op, andOp, andOp,
                                            /*lhs_invert=*/true,
                                            /*rhs_invert=*/false);
    return success();
  }
};

/// Lower a comb::XorOp operation to AIG operations
struct CombXorOpConversion : OpConversionPattern<XorOp> {
  using OpConversionPattern<XorOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(XorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Implement Xor using And with invert flags: a ^ b = (a | b) & (~a | ~b)
    auto a = adaptor.getInputs()[0];
    auto b = adaptor.getInputs()[1];

    // a | b = ~(~a & ~b) & 1
    auto orAB = rewriter.create<aig::AndOp>(op.getLoc(), a, b,
                                            /*lhs_invert=*/true,
                                            /*rhs_invert=*/true);
    auto intType = cast<IntegerType>(a.getType());
    auto one = rewriter.create<hw::ConstantOp>(
        op.getLoc(), APInt::getAllOnes(intType.getWidth()));
    orAB = rewriter.create<aig::AndOp>(op.getLoc(), orAB, one,
                                       /*lhs_invert=*/true,
                                       /*rhs_invert=*/false);

    // ~a | ~b = ~(a & b)
    auto orNotANotB = rewriter.create<aig::AndOp>(op.getLoc(), a, b,
                                                  /*lhs_invert=*/false,
                                                  /*rhs_invert=*/false);
    orNotANotB = rewriter.create<aig::AndOp>(op.getLoc(), orNotANotB, one,
                                             /*lhs_invert=*/true,
                                             /*rhs_invert=*/false);

    rewriter.replaceOpWithNewOp<aig::AndOp>(op, orAB, orNotANotB);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Convert Comb to AIG pass
//===----------------------------------------------------------------------===//

namespace {
struct ConvertCombToAIGPass
    : public impl::ConvertCombToAIGBase<ConvertCombToAIGPass> {
  void runOnOperation() override;
};
} // namespace

static void populateCombToAIGConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<CombAndOpConversion, CombOrOpConversion, CombXorOpConversion,
               VariadicOpConversion<AndOp>, VariadicOpConversion<OrOp>,
               VariadicOpConversion<XorOp>, VariadicOpConversion<AddOp>,
               VariadicOpConversion<MulOp>>(patterns.getContext());
}

void ConvertCombToAIGPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addIllegalDialect<comb::CombDialect>();
  target.addLegalDialect<aig::AIGDialect>();

  RewritePatternSet patterns(&getContext());
  populateCombToAIGConversionPatterns(patterns);

  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
    return signalPassFailure();
}
