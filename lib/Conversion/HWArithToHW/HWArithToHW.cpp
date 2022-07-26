//===- HWArithToHW.cpp - HWArith to HW Lowering pass ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main HWArith to HW Lowering Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/HWArithToHW.h"
#include "../PassDetail.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HWArith/HWArithOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace circt;
using namespace hwarith;

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

// Extract a bit range, specified via start bit and width, from a given value.
static Value extractBits(OpBuilder &builder, Location loc, Value value,
                         unsigned startBit, unsigned bitWidth) {
  SmallVector<Value, 1> result;
  builder.createOrFold<comb::ExtractOp>(result, loc, value, startBit, bitWidth);
  return result[0];
}

// Perform the specified bit-extension (either sign- or zero-extension) for a
// given value to a desired target width.
static Value extendTypeWidth(OpBuilder &builder, Location loc, Value value,
                             unsigned targetWidth, bool signExtension) {
  unsigned sourceWidth = value.getType().getIntOrFloatBitWidth();
  unsigned extensionLength = targetWidth - sourceWidth;

  if (extensionLength == 0)
    return value;

  Value extensionBits;
  // https://circt.llvm.org/docs/Dialects/Comb/RationaleComb/#no-complement-negate-zext-sext-operators
  if (signExtension) {
    // Sign extension
    Value highBit = extractBits(builder, loc, value,
                                /*startBit=*/sourceWidth - 1, /*bitWidth=*/1);
    SmallVector<Value, 1> result;
    builder.createOrFold<comb::ReplicateOp>(result, loc, highBit,
                                            extensionLength);
    extensionBits = result[0];
  } else {
    // Zero extension
    extensionBits = builder
                        .create<hw::ConstantOp>(
                            loc, builder.getIntegerType(extensionLength), 0)
                        ->getOpResult(0);
  }
  return builder.create<comb::ConcatOp>(loc, extensionBits, value)
      ->getOpResult(0);
}

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {
struct ConstantOpLowering : public OpConversionPattern<ConstantOp> {
  using OpConversionPattern<ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp constOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<hw::ConstantOp>(constOp,
                                                constOp.getConstantValue());
    return success();
  }
};
} // namespace

namespace {
struct DivOpLowering : public OpConversionPattern<DivOp> {
  using OpConversionPattern<DivOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DivOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto isLhsTypeSigned =
        op.getOperand(0).getType().template cast<IntegerType>().isSigned();
    auto rhsType = op.getOperand(1).getType().template cast<IntegerType>();
    auto targetType = op.result().getType().template cast<IntegerType>();

    // comb.div* needs identical bitwidths for its operands and its result.
    // Hence, we need to calculate the minimal bitwidth that can be used to
    // represent the result as well as the operands without precision or sign
    // loss. The target size only depends on LHS and already handles the edge
    // cases where the bitwidth needs to be increased by 1. Thus, the targetType
    // is good enough for both the result as well as LHS.
    // The bitwidth for RHS is bit tricky. If the RHS is unsigned and we are
    // about to perform a signed division, then we need one additional bit to
    // avoid misinterpretation of RHS as a signed value!
    bool signedDivision = targetType.isSigned();
    unsigned extendSize = std::max(
        targetType.getWidth(),
        rhsType.getWidth() + (signedDivision && !rhsType.isSigned() ? 1 : 0));

    // Extend the operands
    Value lhsValue = extendTypeWidth(rewriter, loc, adaptor.inputs()[0],
                                     extendSize, isLhsTypeSigned);
    Value rhsValue = extendTypeWidth(rewriter, loc, adaptor.inputs()[1],
                                     extendSize, rhsType.isSigned());

    Value divResult;
    if (signedDivision)
      divResult = rewriter.create<comb::DivSOp>(loc, lhsValue, rhsValue)
                      ->getOpResult(0);
    else
      divResult = rewriter.create<comb::DivUOp>(loc, lhsValue, rhsValue)
                      ->getOpResult(0);

    // finally truncate back to the expected result size!
    Value truncateResult = extractBits(rewriter, loc, divResult, /*startBit=*/0,
                                       /*bitWidth=*/targetType.getWidth());
    rewriter.replaceOp(op, truncateResult);

    return success();
  }
};
} // namespace

namespace {
struct CastOpLowering : public OpConversionPattern<CastOp> {
  using OpConversionPattern<CastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto sourceType = op.in().getType().cast<IntegerType>();
    auto sourceWidth = sourceType.getWidth();
    bool isSourceTypeSigned = sourceType.isSigned();
    auto targetWidth = op.out().getType().cast<IntegerType>().getWidth();

    Value replaceValue;
    if (sourceWidth == targetWidth) {
      // the width does not change, we are done here and can directly use the
      // lowering input value
      replaceValue = adaptor.in();
    } else if (sourceWidth < targetWidth) {
      // bit extensions needed, the type of extension required is determined by
      // the source type only!
      replaceValue = extendTypeWidth(rewriter, op.getLoc(), adaptor.in(),
                                     targetWidth, isSourceTypeSigned);
    } else {
      // bit truncation needed
      replaceValue = extractBits(rewriter, op.getLoc(), adaptor.in(),
                                 /*startBit=*/0, /*bitWidth=*/targetWidth);
    }
    rewriter.replaceOp(op, replaceValue);

    return success();
  }
};
} // namespace

// Templated patterns

namespace {
template <class BinOp, class ReplaceOp>
struct BinaryOpLowering : public OpConversionPattern<BinOp> {
  using OpConversionPattern<BinOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<BinOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(BinOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto isLhsTypeSigned =
        op.getOperand(0).getType().template cast<IntegerType>().isSigned();
    auto isRhsTypeSigned =
        op.getOperand(1).getType().template cast<IntegerType>().isSigned();
    auto targetWidth =
        op.result().getType().template cast<IntegerType>().getWidth();

    Value lhsValue = extendTypeWidth(rewriter, loc, adaptor.inputs()[0],
                                     targetWidth, isLhsTypeSigned);
    Value rhsValue = extendTypeWidth(rewriter, loc, adaptor.inputs()[1],
                                     targetWidth, isRhsTypeSigned);
    rewriter.replaceOpWithNewOp<ReplaceOp>(op, lhsValue, rhsValue);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//

namespace {
class HWArithToHWPass : public HWArithToHWBase<HWArithToHWPass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    target.addIllegalDialect<HWArithDialect>();
    target.addLegalDialect<comb::CombDialect, hw::HWDialect>();

    patterns.add<ConstantOpLowering, CastOpLowering,
                 BinaryOpLowering<AddOp, comb::AddOp>,
                 BinaryOpLowering<SubOp, comb::SubOp>,
                 BinaryOpLowering<MulOp, comb::MulOp>, DivOpLowering>(
        patterns.getContext());

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pass initialization
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> circt::createHWArithToHWPass() {
  return std::make_unique<HWArithToHWPass>();
}
