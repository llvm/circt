//===- LowerArcsToFuncs.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Support/Namespace.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arc-lower-arcs-to-funcs"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_LOWERARCSTOFUNCS
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace arc;
using namespace hw;

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {
struct LowerArcsToFuncsPass
    : public arc::impl::LowerArcsToFuncsBase<LowerArcsToFuncsPass> {
  LowerArcsToFuncsPass() = default;
  LowerArcsToFuncsPass(const LowerArcsToFuncsPass &pass)
      : LowerArcsToFuncsPass() {}

  LogicalResult lowerToFuncs();
  void runOnOperation() override;
};

struct DefineOpLowering : public OpConversionPattern<arc::DefineOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arc::DefineOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto func = rewriter.create<mlir::func::FuncOp>(op.getLoc(), op.getName(),
                                                    op.getFunctionType());
    func->setAttr(
        "llvm.linkage",
        LLVM::LinkageAttr::get(getContext(), LLVM::linkage::Linkage::Internal));
    rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
    rewriter.eraseOp(op);
    return success();
  }
};

struct OutputOpLowering : public OpConversionPattern<arc::OutputOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arc::OutputOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOutputs());
    return success();
  }
};

struct CallOpLowering : public OpConversionPattern<arc::CallOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arc::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    SmallVector<Type> newResultTypes;
    if (failed(
            typeConverter->convertTypes(op.getResultTypes(), newResultTypes)))
      return failure();
    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, newResultTypes, op.getArcAttr(), adaptor.getInputs());
    return success();
  }
};

struct StateOpLowering : public OpConversionPattern<arc::StateOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arc::StateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    SmallVector<Type> newResultTypes;
    if (failed(
            typeConverter->convertTypes(op.getResultTypes(), newResultTypes)))
      return failure();
    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, newResultTypes, op.getArcAttr(), adaptor.getInputs());
    return success();
  }
};

struct ReturnOpLowering : public OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

struct FuncCallOpLowering : public OpConversionPattern<func::CallOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(func::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> newResultTypes;
    if (failed(
            typeConverter->convertTypes(op->getResultTypes(), newResultTypes)))
      return failure();
    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, op.getCalleeAttr(), newResultTypes, adaptor.getOperands());
    return success();
  }
};

} // namespace

static bool isArcType(Type type) {
  return type.isa<StorageType>() || type.isa<MemoryType>() ||
         type.isa<StateType>();
}

static bool hasArcType(TypeRange types) {
  return llvm::any_of(types, isArcType);
}

static bool hasArcType(ValueRange values) {
  return hasArcType(values.getTypes());
}

template <typename Op>
static void addGenericLegality(ConversionTarget &target) {
  target.addDynamicallyLegalOp<Op>([](Op op) {
    return !hasArcType(op->getOperands()) && !hasArcType(op->getResults());
  });
}

static void populateLegality(ConversionTarget &target) {
  target.addLegalDialect<mlir::BuiltinDialect>();
  target.addLegalDialect<hw::HWDialect>();
  target.addLegalDialect<comb::CombDialect>();
  target.addLegalDialect<func::FuncDialect>();
  target.addLegalDialect<scf::SCFDialect>();
  target.addLegalDialect<LLVM::LLVMDialect>();

  target.addIllegalOp<arc::DefineOp>();
  target.addIllegalOp<arc::OutputOp>();
  target.addIllegalOp<arc::StateOp>();

  target.addDynamicallyLegalOp<func::FuncOp>([](func::FuncOp op) {
    auto argsConverted = llvm::none_of(op.getBlocks(), [](auto &block) {
      return hasArcType(block.getArguments());
    });
    auto resultsConverted = !hasArcType(op.getResultTypes());
    return argsConverted && resultsConverted;
  });
  addGenericLegality<func::ReturnOp>(target);
  addGenericLegality<func::CallOp>(target);
}

static void populateOpConversion(RewritePatternSet &patterns,
                                 TypeConverter &typeConverter) {
  auto *context = patterns.getContext();
  // clang-format off
  patterns.add<
    CallOpLowering,
    DefineOpLowering,
    OutputOpLowering,
    FuncCallOpLowering,
    ReturnOpLowering,
    StateOpLowering
  >(typeConverter, context);
  // clang-format on

  mlir::populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
      patterns, typeConverter);
}

static void populateTypeConversion(TypeConverter &typeConverter) {
  typeConverter.addConversion([&](StorageType type) {
    return LLVM::LLVMPointerType::get(IntegerType::get(type.getContext(), 8));
  });
  typeConverter.addConversion([&](MemoryType type) {
    return LLVM::LLVMPointerType::get(
        IntegerType::get(type.getContext(), type.getStride() * 8));
  });
  typeConverter.addConversion([&](StateType type) {
    return LLVM::LLVMPointerType::get(
        typeConverter.convertType(type.getType()));
  });
  typeConverter.addConversion([](hw::ArrayType type) { return type; });
  typeConverter.addConversion([](mlir::IntegerType type) { return type; });
}


/// Perform the lowering to Func and SCF.
LogicalResult LowerArcsToFuncsPass::lowerToFuncs() {
  LLVM_DEBUG(llvm::dbgs() << "Lowering arcs to Func/SCF dialects\n");
  ConversionTarget target(getContext());
  TypeConverter converter;
  RewritePatternSet patterns(&getContext());
  populateLegality(target);
  populateTypeConversion(converter);
  populateOpConversion(patterns, converter);
  return applyPartialConversion(getOperation(), target, std::move(patterns));
}

void LowerArcsToFuncsPass::runOnOperation() {
  if (failed(lowerToFuncs()))
    return signalPassFailure();
}

std::unique_ptr<Pass> arc::createLowerArcsToFuncsPass() {
  return std::make_unique<LowerArcsToFuncsPass>();
}
