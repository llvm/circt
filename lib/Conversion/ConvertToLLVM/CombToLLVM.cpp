//===- CombToLLVM.cpp - Comb to LLVM Conversion Patterns ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This defines the Comb to LLVM Operation Conversion Patterns.
//
//===----------------------------------------------------------------------===//

#include "CombToLLVM.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;
using namespace circt;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

namespace {
template <typename SourceOp, typename TargetOp>
class VariadicOpConversion : public ConvertOpToLLVMPattern<SourceOp> {
public:
  using ConvertOpToLLVMPattern<SourceOp>::ConvertOpToLLVMPattern;
  using Super = VariadicOpConversion<SourceOp, TargetOp>;

  LogicalResult
  matchAndRewrite(SourceOp op, ArrayRef<Value> operands,
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
} // namespace

//===----------------------------------------------------------------------===//
// Arithmetic and Logical operation conversions
//===----------------------------------------------------------------------===//

namespace {

using AndOpConversion = VariadicOpConversion<comb::AndOp, LLVM::AndOp>;
using OrOpConversion = VariadicOpConversion<comb::OrOp, LLVM::OrOp>;
using XorOpConversion = VariadicOpConversion<comb::XorOp, LLVM::XOrOp>;

using ShlOpConversion = OneToOneConvertToLLVMPattern<comb::ShlOp, LLVM::ShlOp>;
using ShrUOpConversion =
    OneToOneConvertToLLVMPattern<comb::ShrUOp, LLVM::LShrOp>;
using ShrSOpConversion =
    OneToOneConvertToLLVMPattern<comb::ShrSOp, LLVM::AShrOp>;

using AddOpConversion = VariadicOpConversion<comb::AddOp, LLVM::AddOp>;
using MulOpConversion = VariadicOpConversion<comb::MulOp, LLVM::MulOp>;
using SubOpConversion = OneToOneConvertToLLVMPattern<comb::SubOp, LLVM::SubOp>;

using DivUOpConversion =
    OneToOneConvertToLLVMPattern<comb::DivUOp, LLVM::UDivOp>;
using DivSOpConversion =
    OneToOneConvertToLLVMPattern<comb::DivSOp, LLVM::SDivOp>;

using ModUOpConversion =
    OneToOneConvertToLLVMPattern<comb::ModUOp, LLVM::URemOp>;
using ModSOpConversion =
    OneToOneConvertToLLVMPattern<comb::ModSOp, LLVM::SRemOp>;

using ICmpOpConversion =
    OneToOneConvertToLLVMPattern<comb::ICmpOp, LLVM::ICmpOp>;

} // namespace

//===----------------------------------------------------------------------===//
// Unary operation conversions
//===----------------------------------------------------------------------===//

namespace {
/// Convert a comb::ParityOp to the LLVM dialect.
struct ParityOpConversion : public ConvertOpToLLVMPattern<comb::ParityOp> {
  using ConvertOpToLLVMPattern<comb::ParityOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(comb::ParityOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto popCount = rewriter.create<LLVM::CtPopOp>(op->getLoc(), op.input());
    rewriter.replaceOpWithNewOp<LLVM::TruncOp>(
        op, IntegerType::get(rewriter.getContext(), 1), popCount);

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Integer width modifying operation conversions
//===----------------------------------------------------------------------===//

namespace {

using SExtOpConversion =
    OneToOneConvertToLLVMPattern<comb::SExtOp, LLVM::SExtOp>;

/// Convert a comb::ExtractOp to LLVM dialect.
struct ExtractOpConversion : public ConvertOpToLLVMPattern<comb::ExtractOp> {
  using ConvertOpToLLVMPattern<comb::ExtractOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(comb::ExtractOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    mlir::Value valueToTrunc = op.input();
    mlir::Type type = op.input().getType();

    if (op.lowBit() != 0) {
      mlir::Value amt = rewriter.create<LLVM::ConstantOp>(op->getLoc(), type,
                                                          op.lowBitAttr());
      valueToTrunc =
          rewriter.create<LLVM::LShrOp>(op->getLoc(), type, op.input(), amt);
    }

    rewriter.replaceOpWithNewOp<LLVM::TruncOp>(op, op.result().getType(),
                                               valueToTrunc);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Other operation conversions
//===----------------------------------------------------------------------===//

namespace {

// comb.mux supports any type thus this conversion relies on the type converter
// to be able to convert the type of the operands and result to an LLVM_Type
using MuxOpConversion =
    OneToOneConvertToLLVMPattern<comb::MuxOp, LLVM::SelectOp>;

/// Convert a comb::ConcatOp to the LLVM dialect.
struct ConcatOpConversion : public ConvertOpToLLVMPattern<comb::ConcatOp> {
  using ConvertOpToLLVMPattern<comb::ConcatOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(comb::ConcatOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto numOperands = op->getNumOperands();
    mlir::Type type = op.result().getType();

    unsigned nextInsertion = type.getIntOrFloatBitWidth();
    auto aggregate = rewriter
                         .create<LLVM::ConstantOp>(op->getLoc(), type,
                                                   IntegerAttr::get(type, 0))
                         .res();

    for (unsigned i = 0; i < numOperands; i++) {
      nextInsertion -= op->getOperand(i).getType().getIntOrFloatBitWidth();

      auto nextInsValue = rewriter.create<LLVM::ConstantOp>(
          op->getLoc(), type, IntegerAttr::get(type, nextInsertion));
      auto extended =
          rewriter.create<LLVM::ZExtOp>(op->getLoc(), type, op->getOperand(i));
      auto shifted = rewriter.create<LLVM::ShlOp>(op->getLoc(), type, extended,
                                                  nextInsValue);
      aggregate =
          rewriter.create<LLVM::OrOp>(op->getLoc(), type, aggregate, shifted)
              .res();
    }

    rewriter.replaceOp(op, aggregate);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern registration
//===----------------------------------------------------------------------===//

void circt::populateCombToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                                 RewritePatternSet &patterns) {

  // Arithmetic and Logical operation conversion patterns.
  patterns
      .add<AddOpConversion, SubOpConversion, MulOpConversion, DivUOpConversion,
           DivSOpConversion, ModUOpConversion, ModSOpConversion,
           ICmpOpConversion, AndOpConversion, OrOpConversion, XorOpConversion,
           ShlOpConversion, ShrUOpConversion, ShrSOpConversion>(converter);

  // Unary operation conversion patterns.
  patterns.add<ParityOpConversion>(converter);

  // Intger width modifying operation conversion patterns.
  patterns.add<ExtractOpConversion, SExtOpConversion, ConcatOpConversion>(
      converter);

  // Other operation conversion patterns.
  patterns.add<MuxOpConversion, ConcatOpConversion>(converter);
}
