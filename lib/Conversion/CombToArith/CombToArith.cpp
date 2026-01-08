//===- CombToArith.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/CombToArith.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace circt {
#define GEN_PASS_DEF_CONVERTCOMBTOARITH
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace circt;
using namespace hw;
using namespace comb;
using namespace mlir;
using namespace arith;

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {
/// Lower a comb::ReplicateOp operation to a comb::ConcatOp
struct CombReplicateOpConversion : OpConversionPattern<ReplicateOp> {
  using OpConversionPattern<ReplicateOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReplicateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type inputType = op.getInput().getType();
    if (isa<IntegerType>(inputType) && inputType.getIntOrFloatBitWidth() == 1) {
      Type outType = rewriter.getIntegerType(op.getMultiple());
      rewriter.replaceOpWithNewOp<ExtSIOp>(op, outType, adaptor.getInput());
      return success();
    }

    SmallVector<Value> inputs(op.getMultiple(), adaptor.getInput());
    rewriter.replaceOpWithNewOp<ConcatOp>(op, inputs);
    return success();
  }
};

/// Lower a hw::ConstantOp operation to a arith::ConstantOp
struct HWConstantOpConversion : OpConversionPattern<hw::ConstantOp> {
  using OpConversionPattern<hw::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hw::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, adaptor.getValueAttr());
    return success();
  }
};

/// Lower a comb::ICmpOp operation to a arith::CmpIOp
struct IcmpOpConversion : OpConversionPattern<ICmpOp> {
  using OpConversionPattern<ICmpOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ICmpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    CmpIPredicate pred;
    switch (adaptor.getPredicate()) {
    case ICmpPredicate::cne:
    case ICmpPredicate::wne:
    case ICmpPredicate::ne:
      pred = CmpIPredicate::ne;
      break;
    case ICmpPredicate::ceq:
    case ICmpPredicate::weq:
    case ICmpPredicate::eq:
      pred = CmpIPredicate::eq;
      break;
    case ICmpPredicate::sge:
      pred = CmpIPredicate::sge;
      break;
    case ICmpPredicate::sgt:
      pred = CmpIPredicate::sgt;
      break;
    case ICmpPredicate::sle:
      pred = CmpIPredicate::sle;
      break;
    case ICmpPredicate::slt:
      pred = CmpIPredicate::slt;
      break;
    case ICmpPredicate::uge:
      pred = CmpIPredicate::uge;
      break;
    case ICmpPredicate::ugt:
      pred = CmpIPredicate::ugt;
      break;
    case ICmpPredicate::ule:
      pred = CmpIPredicate::ule;
      break;
    case ICmpPredicate::ult:
      pred = CmpIPredicate::ult;
      break;
    }

    rewriter.replaceOpWithNewOp<CmpIOp>(op, pred, adaptor.getLhs(),
                                        adaptor.getRhs());
    return success();
  }
};

/// Lower a comb::ExtractOp operation to the arith dialect
struct ExtractOpConversion : OpConversionPattern<ExtractOp> {
  using OpConversionPattern<ExtractOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Value lowBit = arith::ConstantOp::create(
        rewriter, op.getLoc(),
        IntegerAttr::get(adaptor.getInput().getType(), adaptor.getLowBit()));
    Value shifted =
        ShRUIOp::create(rewriter, op.getLoc(), adaptor.getInput(), lowBit);
    rewriter.replaceOpWithNewOp<TruncIOp>(op, op.getResult().getType(),
                                          shifted);
    return success();
  }
};

/// Lower a comb::ConcatOp operation to the arith dialect
struct ConcatOpConversion : OpConversionPattern<ConcatOp> {
  using OpConversionPattern<ConcatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type type = op.getResult().getType();
    Location loc = op.getLoc();

    // Handle the trivial case where we have only one operand. The concat is a
    // no-op in this case.
    if (op.getNumOperands() == 1) {
      rewriter.replaceOp(op, adaptor.getOperands().back());
      return success();
    }

    // The operand at the least significant bit position (the one all the way on
    // the right at the highest index) does not need to be shifted and can just
    // be zero-extended to the final bit width.
    Value aggregate =
        rewriter.createOrFold<ExtUIOp>(loc, type, adaptor.getOperands().back());

    // Shift and OR all the other operands onto the aggregate. Skip the last
    // operand because it has already been incorporated into the aggregate.
    unsigned offset = type.getIntOrFloatBitWidth();
    for (auto operand : adaptor.getOperands().drop_back()) {
      offset -= operand.getType().getIntOrFloatBitWidth();
      auto offsetConst = arith::ConstantOp::create(
          rewriter, loc, IntegerAttr::get(type, offset));
      auto extended = rewriter.createOrFold<ExtUIOp>(loc, type, operand);
      auto shifted = rewriter.createOrFold<ShLIOp>(loc, extended, offsetConst);
      aggregate = rewriter.createOrFold<OrIOp>(loc, aggregate, shifted);
    }

    rewriter.replaceOp(op, aggregate);
    return success();
  }
};

/// Lower the two-operand SourceOp to the two-operand TargetOp
template <typename SourceOp, typename TargetOp>
struct BinaryOpConversion : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<TargetOp>(op, op.getResult().getType(),
                                          adaptor.getOperands());
    return success();
  }
};

/// Lowering for division operations that need to special-case zero-value
/// divisors to not run coarser UB than CIRCT defines.
template <typename SourceOp, typename TargetOp>
struct DivOpConversion : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zero = arith::ConstantOp::create(
        rewriter, loc, rewriter.getIntegerAttr(adaptor.getRhs().getType(), 0));
    Value one = arith::ConstantOp::create(
        rewriter, loc, rewriter.getIntegerAttr(adaptor.getRhs().getType(), 1));
    Value isZero = arith::CmpIOp::create(rewriter, loc, CmpIPredicate::eq,
                                         adaptor.getRhs(), zero);
    Value divisor =
        arith::SelectOp::create(rewriter, loc, isZero, one, adaptor.getRhs());
    rewriter.replaceOpWithNewOp<TargetOp>(op, adaptor.getLhs(), divisor);
    return success();
  }
};

/// Lower a comb::ReplicateOp operation to the LLVM dialect.
template <typename SourceOp, typename TargetOp>
struct VariadicOpConversion : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // TODO: building a tree would be better here
    ValueRange operands = adaptor.getOperands();
    Value runner = operands[0];
    for (Value operand :
         llvm::make_range(operands.begin() + 1, operands.end())) {
      runner = TargetOp::create(rewriter, op.getLoc(), runner, operand);
    }
    rewriter.replaceOp(op, runner);
    return success();
  }
};

// Shifts greater than or equal to the width of the lhs are currently
// unspecified in arith and produce poison in LLVM IR. To prevent undefined
// behaviour we handle this case explicitly.

/// Lower the logical shift SourceOp to the logical shift TargetOp
/// Ensure to produce zero for shift amounts greater than or equal to the width
/// of the lhs
template <typename SourceOp, typename TargetOp>
struct LogicalShiftConversion : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    unsigned shifteeWidth =
        hw::type_cast<IntegerType>(adaptor.getLhs().getType())
            .getIntOrFloatBitWidth();
    auto zeroConstOp = arith::ConstantOp::create(
        rewriter, op.getLoc(), IntegerAttr::get(adaptor.getLhs().getType(), 0));
    auto maxShamtConstOp = arith::ConstantOp::create(
        rewriter, op.getLoc(),
        IntegerAttr::get(adaptor.getLhs().getType(), shifteeWidth));
    auto shiftOp = rewriter.createOrFold<TargetOp>(
        op.getLoc(), adaptor.getLhs(), adaptor.getRhs());
    auto isAllZeroOp = rewriter.createOrFold<CmpIOp>(
        op.getLoc(), CmpIPredicate::uge, adaptor.getRhs(),
        maxShamtConstOp.getResult());
    rewriter.replaceOpWithNewOp<SelectOp>(op, isAllZeroOp, zeroConstOp,
                                          shiftOp);
    return success();
  }
};

/// Lower a comb::ShrSOp operation to a (saturating) arith::ShRSIOp
struct ShrSOpConversion : OpConversionPattern<ShrSOp> {
  using OpConversionPattern<ShrSOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ShrSOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    unsigned shifteeWidth =
        hw::type_cast<IntegerType>(adaptor.getLhs().getType())
            .getIntOrFloatBitWidth();
    // Clamp the shift amount to shifteeWidth - 1
    auto maxShamtMinusOneConstOp = arith::ConstantOp::create(
        rewriter, op.getLoc(),
        IntegerAttr::get(adaptor.getLhs().getType(), shifteeWidth - 1));
    auto shamtOp = rewriter.createOrFold<MinUIOp>(op.getLoc(), adaptor.getRhs(),
                                                  maxShamtMinusOneConstOp);
    rewriter.replaceOpWithNewOp<ShRSIOp>(op, adaptor.getLhs(), shamtOp);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Convert Comb to Arith pass
//===----------------------------------------------------------------------===//

namespace {
struct ConvertCombToArithPass
    : public circt::impl::ConvertCombToArithBase<ConvertCombToArithPass> {
  void runOnOperation() override;
};
} // namespace

void circt::populateCombToArithConversionPatterns(
    TypeConverter &converter, mlir::RewritePatternSet &patterns) {
  patterns.add<
      CombReplicateOpConversion, HWConstantOpConversion, IcmpOpConversion,
      ExtractOpConversion, ConcatOpConversion, ShrSOpConversion,
      LogicalShiftConversion<ShlOp, ShLIOp>,
      LogicalShiftConversion<ShrUOp, ShRUIOp>,
      BinaryOpConversion<SubOp, SubIOp>, DivOpConversion<DivSOp, DivSIOp>,
      DivOpConversion<DivUOp, DivUIOp>, DivOpConversion<ModSOp, RemSIOp>,
      DivOpConversion<ModUOp, RemUIOp>, BinaryOpConversion<MuxOp, SelectOp>,
      VariadicOpConversion<AddOp, AddIOp>, VariadicOpConversion<MulOp, MulIOp>,
      VariadicOpConversion<AndOp, AndIOp>, VariadicOpConversion<OrOp, OrIOp>,
      VariadicOpConversion<XorOp, XOrIOp>>(converter, patterns.getContext());
}

void ConvertCombToArithPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addIllegalDialect<comb::CombDialect>();
  target.addIllegalOp<hw::ConstantOp>();
  target.addLegalDialect<ArithDialect>();
  // Arith does not have an operation equivalent to comb.parity. A lowering
  // would result in undesirably complex logic, therefore, we mark it legal
  // here.
  target.addLegalOp<comb::ParityOp>();
  // This pass is intended to rewrite Comb ops into Arith ops. Other dialects
  // (e.g. LLVM) may legitimately be present when this pass is used in custom
  // pipelines. Treat all unknown operations as legal so we don't attempt to
  // fold/legalize unrelated ops.
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  RewritePatternSet patterns(&getContext());
  TypeConverter converter;
  converter.addConversion([](Type type) { return type; });
  // TODO: a pattern for comb.parity
  populateCombToArithConversionPatterns(converter, patterns);

  ConversionConfig config;
  config.allowPatternRollback = false;
  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns), config)))
    signalPassFailure();
}

std::unique_ptr<Pass> circt::createConvertCombToArithPass() {
  return std::make_unique<ConvertCombToArithPass>();
}
