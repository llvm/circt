//===- MapArithToComb.cpp - Arith-to-comb mapping pass ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the MapArithToComb pass.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/LLHDTypes.h"
#include "circt/Dialect/Sim/SimTypes.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace circt {
#define GEN_PASS_DEF_MAPARITHTOCOMBPASS
#include "circt/Transforms/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;

namespace {

static bool isCombMuxPayloadType(Type type) {
  if (hw::isHWValueType(type))
    return true;

  if (isa<FloatType, LLVM::LLVMPointerType, llhd::TimeType,
          sim::DynamicStringType, sim::FormatStringType, sim::QueueType>(type))
    return true;

  if (auto array = hw::type_dyn_cast<hw::ArrayType>(type))
    return isCombMuxPayloadType(array.getElementType());

  if (auto array = hw::type_dyn_cast<hw::UnpackedArrayType>(type))
    return isCombMuxPayloadType(array.getElementType());

  if (auto structType = hw::type_dyn_cast<hw::StructType>(type)) {
    for (auto field : structType.getElements())
      if (!isCombMuxPayloadType(field.type))
        return false;
    return true;
  }

  if (auto unionType = hw::type_dyn_cast<hw::UnionType>(type)) {
    for (auto field : unionType.getElements())
      if (!isCombMuxPayloadType(field.type))
        return false;
    return true;
  }

  return false;
}

// A type converter which legalizes integer types, thus ensuring that vector
// types are illegal.
class MapArithTypeConverter : public mlir::TypeConverter {
public:
  MapArithTypeConverter() {
    addConversion([](Type type) {
      if (hw::isHWValueType(type))
        return type;

      return Type();
    });
    addConversion([](Type type) -> std::optional<Type> {
      if (isCombMuxPayloadType(type))
        return type;

      return std::nullopt;
    });
  }
};

template <typename TFrom, typename TTo, bool cloneAttrs = false>
class OneToOnePattern : public OpConversionPattern<TFrom> {
public:
  using OpConversionPattern<TFrom>::OpConversionPattern;
  using OpAdaptor = typename TFrom::Adaptor;

  LogicalResult
  matchAndRewrite(TFrom op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TTo>(
        op, adaptor.getOperands(),
        cloneAttrs ? op->getAttrs() : ArrayRef<::mlir::NamedAttribute>());
    return success();
  }
};

class ExtSConversionPattern : public OpConversionPattern<arith::ExtSIOp> {
public:
  using OpConversionPattern<arith::ExtSIOp>::OpConversionPattern;
  using OpAdaptor = typename arith::ExtSIOp::Adaptor;

  LogicalResult
  matchAndRewrite(arith::ExtSIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    size_t outWidth = op.getType().getIntOrFloatBitWidth();
    rewriter.replaceOp(
        op, comb::createOrFoldSExt(rewriter, op.getLoc(), op.getOperand(),
                                   rewriter.getIntegerType(outWidth)));
    return success();
  }
};

class ExtZConversionPattern : public OpConversionPattern<arith::ExtUIOp> {
public:
  using OpConversionPattern<arith::ExtUIOp>::OpConversionPattern;
  using OpAdaptor = typename arith::ExtUIOp::Adaptor;

  LogicalResult
  matchAndRewrite(arith::ExtUIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    size_t outWidth = op.getOut().getType().getIntOrFloatBitWidth();
    size_t inWidth = adaptor.getIn().getType().getIntOrFloatBitWidth();

    rewriter.replaceOp(op, comb::ConcatOp::create(
                               rewriter, loc,
                               hw::ConstantOp::create(
                                   rewriter, loc, APInt(outWidth - inWidth, 0)),
                               adaptor.getIn()));
    return success();
  }
};

class TruncateConversionPattern : public OpConversionPattern<arith::TruncIOp> {
public:
  using OpConversionPattern<arith::TruncIOp>::OpConversionPattern;
  using OpAdaptor = typename arith::TruncIOp::Adaptor;

  LogicalResult
  matchAndRewrite(arith::TruncIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    size_t outWidth = op.getType().getIntOrFloatBitWidth();
    rewriter.replaceOpWithNewOp<comb::ExtractOp>(op, adaptor.getIn(), 0,
                                                 outWidth);
    return success();
  }
};

class CompConversionPattern : public OpConversionPattern<arith::CmpIOp> {
public:
  using OpConversionPattern<arith::CmpIOp>::OpConversionPattern;
  using OpAdaptor = typename arith::CmpIOp::Adaptor;

  static comb::ICmpPredicate
  arithToCombPredicate(arith::CmpIPredicate predicate) {
    switch (predicate) {
    case arith::CmpIPredicate::eq:
      return comb::ICmpPredicate::eq;
    case arith::CmpIPredicate::ne:
      return comb::ICmpPredicate::ne;
    case arith::CmpIPredicate::slt:
      return comb::ICmpPredicate::slt;
    case arith::CmpIPredicate::ult:
      return comb::ICmpPredicate::ult;
    case arith::CmpIPredicate::sle:
      return comb::ICmpPredicate::sle;
    case arith::CmpIPredicate::ule:
      return comb::ICmpPredicate::ule;
    case arith::CmpIPredicate::sgt:
      return comb::ICmpPredicate::sgt;
    case arith::CmpIPredicate::ugt:
      return comb::ICmpPredicate::ugt;
    case arith::CmpIPredicate::sge:
      return comb::ICmpPredicate::sge;
    case arith::CmpIPredicate::uge:
      return comb::ICmpPredicate::uge;
    }
    llvm_unreachable("Unknown predicate");
  }

  LogicalResult
  matchAndRewrite(arith::CmpIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<comb::ICmpOp>(
        op, arithToCombPredicate(op.getPredicate()), adaptor.getLhs(),
        adaptor.getRhs());
    return success();
  }
};

struct ConstantConversionPattern
    : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // `hw.constant` only supports integers.
    if (!isa<IntegerType>(op.getType()))
      return failure();

    rewriter.replaceOpWithNewOp<hw::ConstantOp>(
        op, cast<IntegerAttr>(adaptor.getValue()));
    return success();
  }
};

// Pattern for arith min/max operations.
// These lower to: cmp + mux
template <typename ArithOp, comb::ICmpPredicate pred>
class MinMaxConversionPattern : public OpConversionPattern<ArithOp> {
public:
  using OpConversionPattern<ArithOp>::OpConversionPattern;
  using OpAdaptor = typename ArithOp::Adaptor;

  LogicalResult
  matchAndRewrite(ArithOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // max(a, b) = mux(cmp(a, b), a, b)
    // min(a, b) = mux(cmp(a, b), a, b)
    auto cmp = comb::ICmpOp::create(rewriter, op.getLoc(), pred,
                                    adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOpWithNewOp<comb::MuxOp>(op, cmp, adaptor.getLhs(),
                                             adaptor.getRhs());
    return success();
  }
};

struct MapArithToCombPass
    : public circt::impl::MapArithToCombPassBase<MapArithToCombPass> {
public:
  MapArithToCombPass(bool enableBestEffortLowering) {
    this->enableBestEffortLowering = enableBestEffortLowering;
  }

  void runOnOperation() override {
    auto *ctx = &getContext();

    ConversionTarget target(*ctx);
    target.addLegalDialect<comb::CombDialect, hw::HWDialect>();
    if (!enableBestEffortLowering) {
      target.addIllegalDialect<arith::ArithDialect>();
    } else {
      // We make all arith operations with a potential lowering here (as
      // specified in circt::populateArithToCombPatterns) illegal
      target.addIllegalOp<arith::AddIOp>();
      target.addIllegalOp<arith::SubIOp>();
      target.addIllegalOp<arith::MulIOp>();
      target.addIllegalOp<arith::DivSIOp>();
      target.addIllegalOp<arith::DivUIOp>();
      target.addIllegalOp<arith::RemSIOp>();
      target.addIllegalOp<arith::RemUIOp>();
      target.addIllegalOp<arith::AndIOp>();
      target.addIllegalOp<arith::OrIOp>();
      target.addIllegalOp<arith::XOrIOp>();
      target.addIllegalOp<arith::ShLIOp>();
      target.addIllegalOp<arith::ShRSIOp>();
      target.addIllegalOp<arith::ShRUIOp>();
      target.addIllegalOp<arith::SelectOp>();
      target.addIllegalOp<arith::ExtSIOp>();
      target.addIllegalOp<arith::ExtUIOp>();
      target.addIllegalOp<arith::TruncIOp>();
      target.addIllegalOp<arith::CmpIOp>();
      target.addIllegalOp<arith::MaxSIOp>();
      target.addIllegalOp<arith::MaxUIOp>();
      target.addIllegalOp<arith::MinSIOp>();
      target.addIllegalOp<arith::MinUIOp>();

      // Force integer constants to be mapped to `hw.constant`.
      target.addDynamicallyLegalOp<arith::ConstantOp>([](Operation *op) {
        return !isa<IntegerType>(op->getResult(0).getType());
      });
    }
    MapArithTypeConverter typeConverter;
    RewritePatternSet patterns(ctx);
    populateArithToCombPatterns(patterns, typeConverter);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void circt::populateArithToCombPatterns(mlir::RewritePatternSet &patterns,
                                        TypeConverter &typeConverter) {
  patterns.insert<
      OneToOnePattern<arith::AddIOp, comb::AddOp>,
      OneToOnePattern<arith::SubIOp, comb::SubOp>,
      OneToOnePattern<arith::MulIOp, comb::MulOp>,
      OneToOnePattern<arith::DivSIOp, comb::DivSOp>,
      OneToOnePattern<arith::DivUIOp, comb::DivUOp>,
      OneToOnePattern<arith::RemSIOp, comb::ModSOp>,
      OneToOnePattern<arith::RemUIOp, comb::ModUOp>,
      OneToOnePattern<arith::AndIOp, comb::AndOp>,
      OneToOnePattern<arith::OrIOp, comb::OrOp>,
      OneToOnePattern<arith::XOrIOp, comb::XorOp>,
      OneToOnePattern<arith::ShLIOp, comb::ShlOp>,
      OneToOnePattern<arith::ShRSIOp, comb::ShrSOp>,
      OneToOnePattern<arith::ShRUIOp, comb::ShrUOp>,
      OneToOnePattern<arith::SelectOp, comb::MuxOp>,
      MinMaxConversionPattern<arith::MaxSIOp, comb::ICmpPredicate::sge>,
      MinMaxConversionPattern<arith::MaxUIOp, comb::ICmpPredicate::uge>,
      MinMaxConversionPattern<arith::MinSIOp, comb::ICmpPredicate::sle>,
      MinMaxConversionPattern<arith::MinUIOp, comb::ICmpPredicate::ule>,
      ExtSConversionPattern, ExtZConversionPattern, TruncateConversionPattern,
      CompConversionPattern, ConstantConversionPattern>(typeConverter,
                                                        patterns.getContext());
}

std::unique_ptr<mlir::Pass>
circt::createMapArithToCombPass(bool enableBestEffortLowering) {
  return std::make_unique<MapArithToCombPass>(enableBestEffortLowering);
}
