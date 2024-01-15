//===- CombToSMT.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/CombToSMT.h"
#include "circt/Conversion/HWToSMT.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/SMT/SMTOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace circt {
#define GEN_PASS_DEF_CONVERTCOMBTOSMT
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace circt;
using namespace comb;

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {
/// Lower a comb::ReplicateOp operation to smt::RepeatOp
struct CombReplicateOpConversion : OpConversionPattern<ReplicateOp> {
  using OpConversionPattern<ReplicateOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReplicateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<smt::RepeatOp>(op, op.getMultiple(),
                                               adaptor.getInput());
    return success();
  }
};

/// Lower a comb::ICmpOp operation to a smt::BVCmpOp, smt::EqOp or
/// smt::DistinctOp
struct IcmpOpConversion : OpConversionPattern<ICmpOp> {
  using OpConversionPattern<ICmpOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ICmpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getPredicate() == ICmpPredicate::weq ||
        adaptor.getPredicate() == ICmpPredicate::ceq ||
        adaptor.getPredicate() == ICmpPredicate::wne ||
        adaptor.getPredicate() == ICmpPredicate::cne)
      return rewriter.notifyMatchFailure(op,
                                         "comparison predicate not supported");

    if (adaptor.getPredicate() == ICmpPredicate::eq) {
      rewriter.replaceOpWithNewOp<smt::EqOp>(op, adaptor.getLhs(),
                                             adaptor.getRhs());
      return success();
    }

    if (adaptor.getPredicate() == ICmpPredicate::ne) {
      rewriter.replaceOpWithNewOp<smt::DistinctOp>(op, adaptor.getLhs(),
                                                   adaptor.getRhs());
      return success();
    }

    smt::BVCmpPredicate pred;
    switch (adaptor.getPredicate()) {
    case ICmpPredicate::sge:
      pred = smt::BVCmpPredicate::sge;
      break;
    case ICmpPredicate::sgt:
      pred = smt::BVCmpPredicate::sgt;
      break;
    case ICmpPredicate::sle:
      pred = smt::BVCmpPredicate::sle;
      break;
    case ICmpPredicate::slt:
      pred = smt::BVCmpPredicate::slt;
      break;
    case ICmpPredicate::uge:
      pred = smt::BVCmpPredicate::uge;
      break;
    case ICmpPredicate::ugt:
      pred = smt::BVCmpPredicate::ugt;
      break;
    case ICmpPredicate::ule:
      pred = smt::BVCmpPredicate::ule;
      break;
    case ICmpPredicate::ult:
      pred = smt::BVCmpPredicate::ult;
      break;
    default:
      llvm_unreachable("all cases handled above");
    }

    rewriter.replaceOpWithNewOp<smt::BVCmpOp>(op, pred, adaptor.getLhs(),
                                              adaptor.getRhs());
    return success();
  }
};

/// Lower a comb::ExtractOp operation to an smt::ExtractOp
struct ExtractOpConversion : OpConversionPattern<ExtractOp> {
  using OpConversionPattern<ExtractOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<smt::ExtractOp>(
        op, typeConverter->convertType(op.getResult().getType()),
        adaptor.getLowBitAttr(), adaptor.getInput());
    return success();
  }
};

/// Lower a comb::MuxOp operation to an smt::IteOp
struct MuxOpConversion : OpConversionPattern<MuxOp> {
  using OpConversionPattern<MuxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MuxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value condition = typeConverter->materializeTargetConversion(
        rewriter, op.getLoc(), smt::BoolType::get(getContext()),
        adaptor.getCond());
    rewriter.replaceOpWithNewOp<smt::IteOp>(
        op, condition, adaptor.getTrueValue(), adaptor.getFalseValue());
    return success();
  }
};

/// Lower a comb::SubOp operation to an smt::BVNegOp + smt::BVAddOp
struct SubOpConversion : OpConversionPattern<SubOp> {
  using OpConversionPattern<SubOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SubOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value negRhs = rewriter.create<smt::BVNegOp>(op.getLoc(), adaptor.getRhs());
    rewriter.replaceOpWithNewOp<smt::BVAddOp>(op, adaptor.getLhs(), negRhs);
    return success();
  }
};

/// Lower the SourceOp to the TargetOp one-to-one.
template <typename SourceOp, typename TargetOp>
struct OneToOneOpConversion : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<TargetOp>(
        op,
        OpConversionPattern<SourceOp>::typeConverter->convertType(
            op.getResult().getType()),
        adaptor.getOperands());
    return success();
  }
};

/// Converts an operation with a variadic number of operands to a chain of
/// binary operations assuming left-associativity of the operation.
template <typename SourceOp, typename TargetOp>
struct VariadicToBinaryOpConversion : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ValueRange operands = adaptor.getOperands();
    if (operands.size() < 2)
      return failure();

    Value runner = operands[0];
    for (Value operand : operands.drop_front())
      runner = rewriter.create<TargetOp>(op.getLoc(), runner, operand);

    rewriter.replaceOp(op, runner);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Convert Comb to SMT pass
//===----------------------------------------------------------------------===//

namespace {
struct ConvertCombToSMTPass
    : public impl::ConvertCombToSMTBase<ConvertCombToSMTPass> {
  void runOnOperation() override;
};
} // namespace

void circt::populateCombToSMTConversionPatterns(TypeConverter &converter,
                                                RewritePatternSet &patterns) {
  patterns.add<CombReplicateOpConversion, IcmpOpConversion, ExtractOpConversion,
               SubOpConversion, MuxOpConversion,
               OneToOneOpConversion<ShlOp, smt::BVShlOp>,
               OneToOneOpConversion<ShrUOp, smt::BVLShrOp>,
               OneToOneOpConversion<ShrSOp, smt::BVAShrOp>,
               OneToOneOpConversion<DivSOp, smt::BVSDivOp>,
               OneToOneOpConversion<DivUOp, smt::BVUDivOp>,
               OneToOneOpConversion<ModSOp, smt::BVSRemOp>,
               OneToOneOpConversion<ModUOp, smt::BVURemOp>,
               VariadicToBinaryOpConversion<ConcatOp, smt::ConcatOp>,
               VariadicToBinaryOpConversion<AddOp, smt::BVAddOp>,
               VariadicToBinaryOpConversion<MulOp, smt::BVMulOp>,
               VariadicToBinaryOpConversion<AndOp, smt::BVAndOp>,
               VariadicToBinaryOpConversion<OrOp, smt::BVOrOp>,
               VariadicToBinaryOpConversion<XorOp, smt::BVXOrOp>>(
      converter, patterns.getContext());

  // TODO: there are two unsupported operations in the comb dialect: 'parity'
  // and 'truth_table'.
}

void ConvertCombToSMTPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addIllegalDialect<comb::CombDialect>();
  target.addLegalDialect<smt::SMTDialect>();

  RewritePatternSet patterns(&getContext());
  TypeConverter converter;
  populateHWToSMTTypeConverter(converter);
  // Also add HW patterns because some 'comb' canonicalizers produce constant
  // operations, i.e., even if there is absolutely no HW operation present
  // initially, we might have to convert one.
  populateHWToSMTConversionPatterns(converter, patterns);
  populateCombToSMTConversionPatterns(converter, patterns);

  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
    return signalPassFailure();
}
