//===- CombToAIG.cpp - Comb to AIG Conversion Pass --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main Comb to AIG Conversion Pass Implementation.
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

/// Lower a comb::AndOp operation to aig::AndInverterOp
struct CombAndOpConversion : OpConversionPattern<AndOp> {
  using OpConversionPattern<AndOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AndOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<bool> nonInverts(adaptor.getInputs().size(), false);
    rewriter.replaceOpWithNewOp<aig::AndInverterOp>(op, adaptor.getInputs(),
                                                    nonInverts);
    return success();
  }
};

/// Lower a comb::OrOp operation to aig::AndInverterOp with invert flags
struct CombOrOpConversion : OpConversionPattern<OrOp> {
  using OpConversionPattern<OrOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Implement Or using And and invert flags: a | b = ~(~a & ~b)
    SmallVector<bool> allInverts(adaptor.getInputs().size(), true);
    auto andOp = rewriter.create<aig::AndInverterOp>(
        op.getLoc(), adaptor.getInputs(), allInverts);
    rewriter.replaceOpWithNewOp<aig::AndInverterOp>(op, andOp,
                                                    /*invert=*/true);
    return success();
  }
};

/// Lower a comb::XorOp operation to AIG operations
struct CombXorOpConversion : OpConversionPattern<XorOp> {
  using OpConversionPattern<XorOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(XorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getNumOperands() != 2)
      return failure();
    // Xor using And with invert flags: a ^ b = (a | b) & (~a | ~b)

    // (a | b) = ~(~a & ~b)
    // (~a | ~b) = ~(a & b)
    auto inputs = adaptor.getInputs();
    SmallVector<bool> allInverts(inputs.size(), true);
    SmallVector<bool> allNotInverts(inputs.size(), false);

    auto notAAndNotB =
        rewriter.create<aig::AndInverterOp>(op.getLoc(), inputs, allInverts);
    auto aAndB =
        rewriter.create<aig::AndInverterOp>(op.getLoc(), inputs, allNotInverts);

    rewriter.replaceOpWithNewOp<aig::AndInverterOp>(op, notAAndNotB, aAndB,
                                                    /*lhs_invert=*/true,
                                                    /*rhs_invert=*/true);
    return success();
  }
};

template <typename OpTy>
struct CombLowerVariadicOp : OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<OpTy>::OpAdaptor;
  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto result = lowerFullyAssociativeOp(op, op.getOperands(), rewriter);
    rewriter.replaceOp(op, result);
    return success();
  }

  static Value lowerFullyAssociativeOp(OpTy op, OperandRange operands,
                                       ConversionPatternRewriter &rewriter) {
    Value lhs, rhs;
    switch (operands.size()) {
    case 0:
      assert(false && "cannot be called with empty operand range");
      break;
    case 1:
      return operands[0];
    case 2:
      lhs = operands[0];
      rhs = operands[1];
      return rewriter.create<OpTy>(op.getLoc(), ValueRange{lhs, rhs}, true);
    default:
      auto firstHalf = operands.size() / 2;
      lhs =
          lowerFullyAssociativeOp(op, operands.take_front(firstHalf), rewriter);
      rhs =
          lowerFullyAssociativeOp(op, operands.drop_front(firstHalf), rewriter);
      return rewriter.create<OpTy>(op.getLoc(), ValueRange{lhs, rhs}, true);
    }
  }
};

// Lower comb::MuxOp to AIG operations.
struct CombMuxOpConversion : OpConversionPattern<MuxOp> {
  using OpConversionPattern<MuxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MuxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Implement: c ? a : b = (replicate(c) & a) | (~replicate(c) & b)

    Value cond = op.getCond();
    auto trueVal = op.getTrueValue();
    auto falseVal = op.getFalseValue();

    if (!op.getType().isInteger()) {
      // If the type of the mux is not integer, bitcast the operands first.
      auto widthType = rewriter.getIntegerType(hw::getBitWidth(op.getType()));
      trueVal =
          rewriter.create<hw::BitcastOp>(op->getLoc(), widthType, trueVal);
      falseVal =
          rewriter.create<hw::BitcastOp>(op->getLoc(), widthType, falseVal);
    }

    // Replicate condition if needed
    if (!trueVal.getType().isInteger(1))
      cond = rewriter.create<comb::ReplicateOp>(op.getLoc(), trueVal.getType(),
                                                cond);

    // c ? a : b => (replicate(c) & a) | (~replicate(c) & b)
    auto lhs = rewriter.create<aig::AndInverterOp>(op.getLoc(), cond, trueVal);
    auto rhs = rewriter.create<aig::AndInverterOp>(op.getLoc(), cond, falseVal,
                                                   true, false);

    Value result = rewriter.create<comb::OrOp>(op.getLoc(), lhs, rhs);
    // Insert the bitcast if the type of the mux is not integer.
    if (result.getType() != op.getType())
      result =
          rewriter.create<hw::BitcastOp>(op.getLoc(), op.getType(), result);
    rewriter.replaceOp(op, result);
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
  patterns.add<
      // Bitwise Logical Ops
      CombAndOpConversion, CombOrOpConversion, CombXorOpConversion,
      CombMuxOpConversion,
      // Variadic ops that must be lowered to binary operations
      CombLowerVariadicOp<XorOp>>(patterns.getContext());
}

void ConvertCombToAIGPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addIllegalDialect<comb::CombDialect>();
  // Keep data movement operations like Extract, Concat and Replicate.
  target.addLegalOp<comb::ExtractOp, comb::ConcatOp, comb::ReplicateOp,
                    hw::BitcastOp>();
  target.addLegalDialect<aig::AIGDialect>();

  RewritePatternSet patterns(&getContext());
  populateCombToAIGConversionPatterns(patterns);

  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
    return signalPassFailure();
}
