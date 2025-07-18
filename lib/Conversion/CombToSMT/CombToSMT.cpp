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
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace circt {
#define GEN_PASS_DEF_CONVERTCOMBTOSMT
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
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

    Value result;
    if (adaptor.getPredicate() == ICmpPredicate::eq) {
      result = rewriter.create<smt::EqOp>(op.getLoc(), adaptor.getLhs(),
                                          adaptor.getRhs());
    } else if (adaptor.getPredicate() == ICmpPredicate::ne) {
      result = rewriter.create<smt::DistinctOp>(op.getLoc(), adaptor.getLhs(),
                                                adaptor.getRhs());
    } else {
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

      result = rewriter.create<smt::BVCmpOp>(
          op.getLoc(), pred, adaptor.getLhs(), adaptor.getRhs());
    }

    Value convVal = typeConverter->materializeTargetConversion(
        rewriter, op.getLoc(), typeConverter->convertType(op.getType()),
        result);
    if (!convVal)
      return failure();

    rewriter.replaceOp(op, convVal);
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

/// Lower a comb::ParityOp operation to a chain of smt::Extract + XOr ops
struct ParityOpConversion : OpConversionPattern<ParityOp> {
  using OpConversionPattern<ParityOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ParityOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    unsigned bitwidth =
        cast<smt::BitVectorType>(adaptor.getInput().getType()).getWidth();

    // Note: the SMT bitvector type does not support 0 bitwidth vectors and thus
    // the type conversion should already fail.
    Type oneBitTy = smt::BitVectorType::get(getContext(), 1);
    Value runner =
        rewriter.create<smt::ExtractOp>(loc, oneBitTy, 0, adaptor.getInput());
    for (unsigned i = 1; i < bitwidth; ++i) {
      Value ext =
          rewriter.create<smt::ExtractOp>(loc, oneBitTy, i, adaptor.getInput());
      runner = rewriter.create<smt::BVXOrOp>(loc, runner, ext);
    }

    rewriter.replaceOp(op, runner);
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

/// Lower the SourceOp to the TargetOp special-casing if the second operand is
/// zero to return a new symbolic value.
template <typename SourceOp, typename TargetOp>
struct DivisionOpConversion : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto type = dyn_cast<smt::BitVectorType>(adaptor.getRhs().getType());
    if (!type)
      return failure();

    auto resultType = OpConversionPattern<SourceOp>::typeConverter->convertType(
        op.getResult().getType());
    Value zero =
        rewriter.create<smt::BVConstantOp>(loc, APInt(type.getWidth(), 0));
    Value isZero = rewriter.create<smt::EqOp>(loc, adaptor.getRhs(), zero);
    Value symbolicVal = rewriter.create<smt::DeclareFunOp>(loc, resultType);
    Value division =
        rewriter.create<TargetOp>(loc, resultType, adaptor.getOperands());
    rewriter.replaceOpWithNewOp<smt::IteOp>(op, isZero, symbolicVal, division);
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
    : public circt::impl::ConvertCombToSMTBase<ConvertCombToSMTPass> {
  void runOnOperation() override;
};
} // namespace

void circt::populateCombToSMTConversionPatterns(TypeConverter &converter,
                                                RewritePatternSet &patterns) {
  patterns.add<CombReplicateOpConversion, IcmpOpConversion, ExtractOpConversion,
               SubOpConversion, MuxOpConversion, ParityOpConversion,
               OneToOneOpConversion<ShlOp, smt::BVShlOp>,
               OneToOneOpConversion<ShrUOp, smt::BVLShrOp>,
               OneToOneOpConversion<ShrSOp, smt::BVAShrOp>,
               DivisionOpConversion<DivSOp, smt::BVSDivOp>,
               DivisionOpConversion<DivUOp, smt::BVUDivOp>,
               DivisionOpConversion<ModSOp, smt::BVSRemOp>,
               DivisionOpConversion<ModUOp, smt::BVURemOp>,
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
  target.addIllegalDialect<hw::HWDialect>();
  target.addIllegalOp<seq::FromClockOp>();
  target.addIllegalOp<seq::ToClockOp>();
  target.addIllegalDialect<comb::CombDialect>();
  target.addLegalDialect<smt::SMTDialect>();
  target.addLegalDialect<mlir::func::FuncDialect>();

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
