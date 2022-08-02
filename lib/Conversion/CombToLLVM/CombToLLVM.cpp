//===- CombToLLVM.cpp - Comb to LLVM Conversion Pass ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main Comb to LLVM Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/CombToLLVM.h"
#include "../PassDetail.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace circt;

//===----------------------------------------------------------------------===//
// Extraction operation conversions
//===----------------------------------------------------------------------===//

namespace {
/// Convert a comb::ExtractOp to LLVM dialect.
struct CombExtractOpConversion : public ConvertToLLVMPattern {
  explicit CombExtractOpConversion(MLIRContext *ctx,
                                   LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(comb::ExtractOp::getOperationName(), ctx,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto extractOp = cast<comb::ExtractOp>(op);
    mlir::Value valueToTrunc = extractOp.getInput();
    mlir::Type type = extractOp.getInput().getType();

    if (extractOp.getLowBit() != 0) {
      mlir::Value amt = rewriter.create<LLVM::ConstantOp>(
          op->getLoc(), type, extractOp.getLowBitAttr());
      valueToTrunc = rewriter.create<LLVM::LShrOp>(op->getLoc(), type,
                                                   extractOp.getInput(), amt);
    }

    rewriter.replaceOpWithNewOp<LLVM::TruncOp>(
        op, extractOp.getResult().getType(), valueToTrunc);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Concat operations conversion
//===----------------------------------------------------------------------===//

namespace {
/// Convert a comb::ConcatOp to the LLVM dialect.
struct CombConcatOpConversion : public ConvertToLLVMPattern {
  explicit CombConcatOpConversion(MLIRContext *ctx,
                                  LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(comb::ConcatOp::getOperationName(), ctx,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto concatOp = cast<comb::ConcatOp>(op);
    auto numOperands = concatOp->getNumOperands();
    mlir::Type type = concatOp.getResult().getType();

    unsigned nextInsertion = type.getIntOrFloatBitWidth();
    auto aggregate = rewriter
                         .create<LLVM::ConstantOp>(op->getLoc(), type,
                                                   IntegerAttr::get(type, 0))
                         .getRes();

    for (unsigned i = 0; i < numOperands; i++) {
      nextInsertion -=
          concatOp->getOperand(i).getType().getIntOrFloatBitWidth();

      auto nextInsValue = rewriter.create<LLVM::ConstantOp>(
          op->getLoc(), type, IntegerAttr::get(type, nextInsertion));
      auto extended = rewriter.create<LLVM::ZExtOp>(op->getLoc(), type,
                                                    concatOp->getOperand(i));
      auto shifted = rewriter.create<LLVM::ShlOp>(op->getLoc(), type, extended,
                                                  nextInsValue);
      aggregate =
          rewriter.create<LLVM::OrOp>(op->getLoc(), type, aggregate, shifted)
              .getRes();
    }

    rewriter.replaceOp(op, aggregate);
    return success();
  }
};
} // namespace

namespace {
/// Lower a comb::ReplicateOp operation to the LLVM dialect.
struct CombReplicateOpConversion
    : public ConvertOpToLLVMPattern<comb::ReplicateOp> {
  using ConvertOpToLLVMPattern<comb::ReplicateOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(comb::ReplicateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    std::vector<Value> inputs(op.getMultiple(), op.getInput());
    rewriter.replaceOpWithNewOp<comb::ConcatOp>(op, inputs);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Bitwise conversions
//===----------------------------------------------------------------------===//

namespace {
template <typename SourceOp, typename TargetOp>
class VariadicOpConversion : public ConvertOpToLLVMPattern<SourceOp> {
public:
  using OpAdaptor = typename SourceOp::Adaptor;
  using ConvertOpToLLVMPattern<SourceOp>::ConvertOpToLLVMPattern;
  using Super = VariadicOpConversion<SourceOp, TargetOp>;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    size_t numOperands = op.getOperands().size();
    // All operands have the same type.
    Type type = op.getOperandTypes().front();
    auto replacement = op.getOperand(0);

    for (unsigned i = 1; i < numOperands; i++) {
      replacement = rewriter.create<TargetOp>(op.getLoc(), type, replacement,
                                              op.getOperand(i));
    }

    rewriter.replaceOp(op, replacement);

    return success();
  }
};

using AndOpConversion = VariadicOpConversion<comb::AndOp, LLVM::AndOp>;
using OrOpConversion = VariadicOpConversion<comb::OrOp, LLVM::OrOp>;
using XorOpConversion = VariadicOpConversion<comb::XorOp, LLVM::XOrOp>;

using CombShlOpConversion =
    OneToOneConvertToLLVMPattern<comb::ShlOp, LLVM::ShlOp>;
using CombShrUOpConversion =
    OneToOneConvertToLLVMPattern<comb::ShrUOp, LLVM::LShrOp>;
using CombShrSOpConversion =
    OneToOneConvertToLLVMPattern<comb::ShrSOp, LLVM::AShrOp>;

} // namespace

//===----------------------------------------------------------------------===//
// Arithmetic conversions
//===----------------------------------------------------------------------===//

namespace {

using CombAddOpConversion = VariadicOpConversion<comb::AddOp, LLVM::AddOp>;
using CombMulOpConversion = VariadicOpConversion<comb::MulOp, LLVM::MulOp>;
using CombSubOpConversion =
    OneToOneConvertToLLVMPattern<comb::SubOp, LLVM::SubOp>;

using CombDivUOpConversion =
    OneToOneConvertToLLVMPattern<comb::DivUOp, LLVM::UDivOp>;
using CombDivSOpConversion =
    OneToOneConvertToLLVMPattern<comb::DivSOp, LLVM::SDivOp>;

using CombModUOpConversion =
    OneToOneConvertToLLVMPattern<comb::ModUOp, LLVM::URemOp>;
using CombModSOpConversion =
    OneToOneConvertToLLVMPattern<comb::ModSOp, LLVM::SRemOp>;

using CombICmpOpConversion =
    OneToOneConvertToLLVMPattern<comb::ICmpOp, LLVM::ICmpOp>;

// comb.mux supports any type thus this conversion relies on the type converter
// to be able to convert the type of the operands and result to an LLVM_Type
using CombMuxOpConversion =
    OneToOneConvertToLLVMPattern<comb::MuxOp, LLVM::SelectOp>;

} // namespace

namespace {
/// Convert a comb::ParityOp to the LLVM dialect.
struct CombParityOpConversion : public ConvertToLLVMPattern {
  explicit CombParityOpConversion(MLIRContext *ctx,
                                  LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(comb::ParityOp::getOperationName(), ctx,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto parityOp = cast<comb::ParityOp>(op);

    auto popCount =
        rewriter.create<LLVM::CtPopOp>(op->getLoc(), parityOp.getInput());
    rewriter.replaceOpWithNewOp<LLVM::TruncOp>(
        op, IntegerType::get(rewriter.getContext(), 1), popCount);

    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass initialization
//===----------------------------------------------------------------------===//

namespace {
struct CombToLLVMLoweringPass
    : public ConvertCombToLLVMBase<CombToLLVMLoweringPass> {
  void runOnOperation() override;
};
} // namespace

void circt::populateCombToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                                 RewritePatternSet &patterns) {
  MLIRContext *ctx = converter.getDialect()->getContext();

  // Extract conversion patterns.
  patterns.add<CombExtractOpConversion, CombConcatOpConversion>(ctx, converter);

  // Bitwise conversion patterns.
  patterns.add<CombParityOpConversion>(ctx, converter);
  patterns.add<AndOpConversion, OrOpConversion, XorOpConversion>(converter);
  patterns.add<CombShlOpConversion, CombShrUOpConversion, CombShrSOpConversion>(
      converter);

  // Arithmetic conversion patterns.
  patterns.add<CombAddOpConversion, CombSubOpConversion, CombMulOpConversion,
               CombDivUOpConversion, CombDivSOpConversion, CombModUOpConversion,
               CombModSOpConversion, CombICmpOpConversion, CombMuxOpConversion,
               CombReplicateOpConversion>(converter);
}

void CombToLLVMLoweringPass::runOnOperation() {

  RewritePatternSet patterns(&getContext());
  auto converter = mlir::LLVMTypeConverter(&getContext());

  LLVMConversionTarget target(getContext());
  target.addLegalOp<UnrealizedConversionCastOp>();
  target.addLegalOp<ModuleOp>();
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addIllegalDialect<comb::CombDialect>();

  // Setup the conversion.
  populateCombToLLVMConversionPatterns(converter, patterns);

  // Apply a partial conversion.
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

/// Create an Comb to LLVM conversion pass.
std::unique_ptr<OperationPass<ModuleOp>> circt::createConvertCombToLLVMPass() {
  return std::make_unique<CombToLLVMLoweringPass>();
}
