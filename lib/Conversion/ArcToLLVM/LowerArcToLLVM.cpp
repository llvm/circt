//===- LowerArcToLLVM.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "circt/Conversion/ArcToLLVM.h"
#include "circt/Conversion/CombToLLVM.h"
#include "circt/Conversion/HWToLLVM.h"
#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/Namespace.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arc-lower-to-llvm"

using namespace mlir;
using namespace circt;
using namespace arc;
using namespace hw;

//===----------------------------------------------------------------------===//
// Lowering Patterns
//===----------------------------------------------------------------------===//

namespace {

struct ModelOpLowering : public OpConversionPattern<arc::ModelOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arc::ModelOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    {
      IRRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(&op.getBodyBlock());
      rewriter.create<func::ReturnOp>(op.getLoc());
    }
    auto funcName = rewriter.getStringAttr(op.getName() + "_eval");
    auto funcType =
        rewriter.getFunctionType(op.getBody().getArgumentTypes(), {});
    auto func =
        rewriter.create<mlir::func::FuncOp>(op.getLoc(), funcName, funcType);
    rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
    rewriter.eraseOp(op);
    return success();
  }
};

struct AllocStorageOpLowering
    : public OpConversionPattern<arc::AllocStorageOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arc::AllocStorageOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto type = typeConverter->convertType(op.getType());
    if (!op.getOffset().has_value())
      return failure();
    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(op, type, rewriter.getI8Type(),
                                             adaptor.getInput(),
                                             LLVM::GEPArg(*op.getOffset()));
    return success();
  }
};

template <class ConcreteOp>
struct AllocStateLikeOpLowering : public OpConversionPattern<ConcreteOp> {
  using OpConversionPattern<ConcreteOp>::OpConversionPattern;
  using OpConversionPattern<ConcreteOp>::typeConverter;
  using OpAdaptor = typename ConcreteOp::Adaptor;

  LogicalResult
  matchAndRewrite(ConcreteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // Get a pointer to the correct offset in the storage.
    auto offsetAttr = op->template getAttrOfType<IntegerAttr>("offset");
    if (!offsetAttr)
      return failure();
    Value ptr = rewriter.create<LLVM::GEPOp>(
        op->getLoc(), adaptor.getStorage().getType(), rewriter.getI8Type(),
        adaptor.getStorage(),
        LLVM::GEPArg(offsetAttr.getValue().getZExtValue()));
    rewriter.replaceOp(op, ptr);
    return success();
  }
};

struct StateReadOpLowering : public OpConversionPattern<arc::StateReadOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arc::StateReadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto type = typeConverter->convertType(op.getType());
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, type, adaptor.getState());
    return success();
  }
};

struct StateWriteOpLowering : public OpConversionPattern<arc::StateWriteOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arc::StateWriteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (adaptor.getCondition()) {
      rewriter.replaceOpWithNewOp<scf::IfOp>(
          op, adaptor.getCondition(), [&](auto &builder, auto loc) {
            builder.template create<LLVM::StoreOp>(loc, adaptor.getValue(),
                                                   adaptor.getState());
            builder.template create<scf::YieldOp>(loc);
          });
    } else {
      rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, adaptor.getValue(),
                                                 adaptor.getState());
    }
    return success();
  }
};

struct AllocMemoryOpLowering : public OpConversionPattern<arc::AllocMemoryOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arc::AllocMemoryOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto offsetAttr = op->getAttrOfType<IntegerAttr>("offset");
    if (!offsetAttr)
      return failure();
    Value ptr = rewriter.create<LLVM::GEPOp>(
        op.getLoc(), adaptor.getStorage().getType(), rewriter.getI8Type(),
        adaptor.getStorage(),
        LLVM::GEPArg(offsetAttr.getValue().getZExtValue()));

    rewriter.replaceOp(op, ptr);
    return success();
  }
};

struct StorageGetOpLowering : public OpConversionPattern<arc::StorageGetOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arc::StorageGetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Value offset = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), rewriter.getI32Type(), op.getOffsetAttr());
    Value ptr = rewriter.create<LLVM::GEPOp>(
        op.getLoc(), adaptor.getStorage().getType(), rewriter.getI8Type(),
        adaptor.getStorage(), offset);
    rewriter.replaceOp(op, ptr);
    return success();
  }
};

struct MemoryAccess {
  Value ptr;
  Value withinBounds;
};

static MemoryAccess prepareMemoryAccess(Location loc, Value memory,
                                        Value address, MemoryType type,
                                        ConversionPatternRewriter &rewriter) {
  auto zextAddrType = rewriter.getIntegerType(
      address.getType().cast<IntegerType>().getWidth() + 1);
  Value addr = rewriter.create<LLVM::ZExtOp>(loc, zextAddrType, address);
  Value addrLimit = rewriter.create<LLVM::ConstantOp>(
      loc, zextAddrType, rewriter.getI32IntegerAttr(type.getNumWords()));
  Value withinBounds = rewriter.create<LLVM::ICmpOp>(
      loc, LLVM::ICmpPredicate::ult, addr, addrLimit);
  Value ptr = rewriter.create<LLVM::GEPOp>(
      loc, LLVM::LLVMPointerType::get(memory.getContext()),
      rewriter.getIntegerType(type.getStride() * 8), memory, ValueRange{addr});
  return {ptr, withinBounds};
}

struct MemoryReadOpLowering : public OpConversionPattern<arc::MemoryReadOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arc::MemoryReadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto type = typeConverter->convertType(op.getType());
    auto memoryType = op.getMemory().getType().cast<MemoryType>();
    auto access =
        prepareMemoryAccess(op.getLoc(), adaptor.getMemory(),
                            adaptor.getAddress(), memoryType, rewriter);

    // Only attempt to read the memory if the address is within bounds,
    // otherwise produce a zero value.
    rewriter.replaceOpWithNewOp<scf::IfOp>(
        op, access.withinBounds,
        [&](auto &builder, auto loc) {
          Value loadOp = builder.template create<LLVM::LoadOp>(
              loc, memoryType.getWordType(), access.ptr);
          builder.template create<scf::YieldOp>(loc, loadOp);
        },
        [&](auto &builder, auto loc) {
          Value zeroValue = builder.template create<LLVM::ConstantOp>(
              loc, type, builder.getI64IntegerAttr(0));
          builder.template create<scf::YieldOp>(loc, zeroValue);
        });
    return success();
  }
};

struct MemoryWriteOpLowering : public OpConversionPattern<arc::MemoryWriteOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arc::MemoryWriteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto access = prepareMemoryAccess(
        op.getLoc(), adaptor.getMemory(), adaptor.getAddress(),
        op.getMemory().getType().cast<MemoryType>(), rewriter);
    auto enable = access.withinBounds;
    if (adaptor.getEnable())
      enable = rewriter.create<LLVM::AndOp>(op.getLoc(), adaptor.getEnable(),
                                            enable);

    // Only attempt to write the memory if the address is within bounds.
    rewriter.replaceOpWithNewOp<scf::IfOp>(
        op, enable, [&](auto &builder, auto loc) {
          builder.template create<LLVM::StoreOp>(loc, adaptor.getData(),
                                                 access.ptr);
          builder.template create<scf::YieldOp>(loc);
        });
    return success();
  }
};

/// A dummy lowering for clock gates to an AND gate.
struct ClockGateOpLowering : public OpConversionPattern<arc::ClockGateOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arc::ClockGateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<comb::AndOp>(op, adaptor.getInput(),
                                             adaptor.getEnable(), true);
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

struct ZeroCountOpLowering : public OpConversionPattern<arc::ZeroCountOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arc::ZeroCountOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Use poison when input is zero.
    IntegerAttr isZeroPoison = rewriter.getBoolAttr(true);

    if (op.getPredicate() == arc::ZeroCountPredicate::leading) {
      rewriter.replaceOpWithNewOp<LLVM::CountLeadingZerosOp>(
          op, adaptor.getInput().getType(), adaptor.getInput(), isZeroPoison);
      return success();
    }

    rewriter.replaceOpWithNewOp<LLVM::CountTrailingZerosOp>(
        op, adaptor.getInput().getType(), adaptor.getInput(), isZeroPoison);
    return success();
  }
};

template <typename OpTy>
struct ReplaceOpWithInputPattern : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpTy::Adaptor;
  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

} // namespace

static bool isArcType(Type type) {
  return type.isa<StorageType>() || type.isa<MemoryType>() ||
         type.isa<StateType>() || type.isa<seq::ClockType>();
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
  target.addIllegalDialect<seq::SeqDialect>();

  target.addIllegalOp<arc::DefineOp>();
  target.addIllegalOp<arc::OutputOp>();
  target.addIllegalOp<arc::StateOp>();
  target.addIllegalOp<arc::ClockTreeOp>();
  target.addIllegalOp<arc::PassThroughOp>();

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

static void populateTypeConversion(TypeConverter &typeConverter) {
  typeConverter.addConversion([&](seq::ClockType type) {
    return IntegerType::get(type.getContext(), 1);
  });
  typeConverter.addConversion([&](StorageType type) {
    return LLVM::LLVMPointerType::get(type.getContext());
  });
  typeConverter.addConversion([&](MemoryType type) {
    return LLVM::LLVMPointerType::get(type.getContext());
  });
  typeConverter.addConversion([&](StateType type) {
    return LLVM::LLVMPointerType::get(type.getContext());
  });
  typeConverter.addConversion([](hw::ArrayType type) { return type; });
  typeConverter.addConversion([](mlir::IntegerType type) { return type; });
}

static void populateOpConversion(RewritePatternSet &patterns,
                                 TypeConverter &typeConverter) {
  auto *context = patterns.getContext();
  // clang-format off
  patterns.add<
    AllocMemoryOpLowering,
    AllocStateLikeOpLowering<arc::AllocStateOp>,
    AllocStateLikeOpLowering<arc::RootInputOp>,
    AllocStateLikeOpLowering<arc::RootOutputOp>,
    AllocStorageOpLowering,
    ClockGateOpLowering,
    FuncCallOpLowering,
    MemoryReadOpLowering,
    MemoryWriteOpLowering,
    ModelOpLowering,
    ReplaceOpWithInputPattern<seq::ToClockOp>,
    ReplaceOpWithInputPattern<seq::FromClockOp>,
    ReturnOpLowering,
    StateReadOpLowering,
    StateWriteOpLowering,
    StorageGetOpLowering,
    ZeroCountOpLowering
  >(typeConverter, context);
  // clang-format on

  mlir::populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
      patterns, typeConverter);
}

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {
struct LowerArcToLLVMPass : public LowerArcToLLVMBase<LowerArcToLLVMPass> {
  void runOnOperation() override;
  LogicalResult lowerToMLIR();
  LogicalResult lowerArcToLLVM();
};
} // namespace

void LowerArcToLLVMPass::runOnOperation() {
  if (failed(lowerToMLIR()))
    return signalPassFailure();

  if (failed(lowerArcToLLVM()))
    return signalPassFailure();
}

/// Perform the lowering to Func and SCF.
LogicalResult LowerArcToLLVMPass::lowerToMLIR() {
  LLVM_DEBUG(llvm::dbgs() << "Lowering arcs to Func/SCF dialects\n");
  ConversionTarget target(getContext());
  TypeConverter converter;
  RewritePatternSet patterns(&getContext());
  populateLegality(target);
  populateTypeConversion(converter);
  populateOpConversion(patterns, converter);
  return applyPartialConversion(getOperation(), target, std::move(patterns));
}

/// Perform lowering to LLVM.
LogicalResult LowerArcToLLVMPass::lowerArcToLLVM() {
  LLVM_DEBUG(llvm::dbgs() << "Lowering to LLVM dialect\n");

  Namespace globals;
  SymbolCache cache;
  cache.addDefinitions(getOperation());
  globals.add(cache);

  LLVMConversionTarget target(getContext());
  LLVMTypeConverter converter(&getContext());
  RewritePatternSet patterns(&getContext());
  target.addLegalOp<mlir::ModuleOp>();
  target.addIllegalOp<arc::ModelOp>();
  populateSCFToControlFlowConversionPatterns(patterns);
  populateFuncToLLVMConversionPatterns(converter, patterns);
  cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);

  DenseMap<std::pair<Type, ArrayAttr>, LLVM::GlobalOp> constAggregateGlobalsMap;
  populateHWToLLVMConversionPatterns(converter, patterns, globals,
                                     constAggregateGlobalsMap);
  populateHWToLLVMTypeConversions(converter);
  populateCombToLLVMConversionPatterns(converter, patterns);
  arith::populateArithToLLVMConversionPatterns(converter, patterns);

  return applyFullConversion(getOperation(), target, std::move(patterns));
}

std::unique_ptr<OperationPass<ModuleOp>> circt::createLowerArcToLLVMPass() {
  return std::make_unique<LowerArcToLLVMPass>();
}
