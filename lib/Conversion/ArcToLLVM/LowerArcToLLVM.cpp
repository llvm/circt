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
#include "circt/Dialect/Arc/ModelInfo.h"
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

#define DEBUG_TYPE "lower-arc-to-llvm"

using namespace mlir;
using namespace circt;
using namespace arc;
using namespace hw;

//===----------------------------------------------------------------------===//
// Lowering Patterns
//===----------------------------------------------------------------------===//

static llvm::Twine evalSymbolFromModelName(StringRef modelName) {
  return modelName + "_eval";
}

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
    auto funcName =
        rewriter.getStringAttr(evalSymbolFromModelName(op.getName()));
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
struct ClockGateOpLowering : public OpConversionPattern<seq::ClockGateOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(seq::ClockGateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<comb::AndOp>(op, adaptor.getInput(),
                                             adaptor.getEnable(), true);
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

//===----------------------------------------------------------------------===//
// Simulation Orchestration Lowering Patterns
//===----------------------------------------------------------------------===//

static Value createPtrToPortState(ConversionPatternRewriter &rewriter,
                                  Location loc, Value state,
                                  const StateInfo *port) {
  MLIRContext *ctx = rewriter.getContext();
  return rewriter.create<LLVM::GEPOp>(loc, LLVM::LLVMPointerType::get(ctx),
                                      IntegerType::get(ctx, 8), state,
                                      LLVM::GEPArg(port->offset));
}

namespace {

struct SimInstantiateLowering
    : public OpConversionPattern<arc::SimInstantiate> {
  SimInstantiateLowering(const TypeConverter &typeConverter,
                         MLIRContext *context, ArrayRef<ModelInfo> modelInfo)
      : OpConversionPattern(typeConverter, context), modelInfo(modelInfo) {}

  LogicalResult
  matchAndRewrite(arc::SimInstantiate op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    const ModelInfo *model = llvm::find_if(
        modelInfo, [&](auto x) { return x.name == op.getType().getModel(); });
    if (model == modelInfo.end()) {
      op.emitError("model not found");
      return failure();
    }

    ConversionPatternRewriter::InsertionGuard guard(rewriter);

    Location loc = op.getLoc();
    Value numStateBytes = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getIntegerType(sizeof(size_t) * 8), model->numStateBytes);
    Value allocated = rewriter.replaceOpWithNewOp<LLVM::AllocaOp>(
        op, LLVM::LLVMPointerType::get(getContext()), rewriter.getI8Type(),
        numStateBytes);
    rewriter.setInsertionPointAfterValue(allocated);
    Value zero =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI8Type(), 0);
    rewriter.create<LLVM::MemsetOp>(loc, allocated, zero, numStateBytes, false);

    return success();
  }

  ArrayRef<ModelInfo> modelInfo;
};

struct SimSetInputLowering : public OpConversionPattern<arc::SimSetInput> {
  SimSetInputLowering(const TypeConverter &typeConverter, MLIRContext *context,
                      ArrayRef<ModelInfo> modelInfo)
      : OpConversionPattern(typeConverter, context), modelInfo(modelInfo) {}

  LogicalResult
  matchAndRewrite(arc::SimSetInput op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    const ModelInfo *model = llvm::find_if(modelInfo, [&](auto x) {
      return x.name ==
             op.getInstance().getType().cast<SimModelInstanceType>().getModel();
    });
    if (model == modelInfo.end())
      return op.emitError("model not found");

    const StateInfo *port = llvm::find_if(
        model->states, [&](auto x) { return x.name == op.getInput(); });
    if (port == model->states.end())
      return op.emitError("input not found on model");

    if (port->numBits != op.getValue().getType().getWidth())
      return op.emitError("expected input of width ")
             << port->numBits << ", got " << op.getValue().getType().getWidth();

    if (port->type != StateInfo::Type::Input)
      return op.emitError("provided port is not an input port");

    Value statePtr = createPtrToPortState(rewriter, op.getLoc(),
                                          adaptor.getInstance(), port);
    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, adaptor.getValue(),
                                               statePtr);

    return success();
  }

  ArrayRef<ModelInfo> modelInfo;
};

struct SimGetPortLowering : public OpConversionPattern<arc::SimGetPort> {
  SimGetPortLowering(const TypeConverter &typeConverter, MLIRContext *context,
                     ArrayRef<ModelInfo> modelInfo)
      : OpConversionPattern(typeConverter, context), modelInfo(modelInfo) {}

  LogicalResult
  matchAndRewrite(arc::SimGetPort op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    const ModelInfo *model = llvm::find_if(modelInfo, [&](auto x) {
      return x.name ==
             op.getInstance().getType().cast<SimModelInstanceType>().getModel();
    });
    if (model == modelInfo.end())
      return op.emitError("model not found");

    const StateInfo *port = llvm::find_if(
        model->states, [&](auto x) { return x.name == op.getPort(); });
    if (port == model->states.end())
      return op.emitError("port not found on model");

    if (port->numBits != op.getValue().getType().getWidth())
      return op.emitError("expected port of width ")
             << port->numBits << ", got " << op.getValue().getType().getWidth();

    Value statePtr = createPtrToPortState(rewriter, op.getLoc(),
                                          adaptor.getInstance(), port);
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, op.getValue().getType(),
                                              statePtr);

    return success();
  }

  ArrayRef<ModelInfo> modelInfo;
};

struct SimStepLowering : public OpConversionPattern<arc::SimStep> {
  SimStepLowering(const TypeConverter &typeConverter, MLIRContext *context,
                  ArrayRef<ModelInfo> modelInfo)
      : OpConversionPattern(typeConverter, context), modelInfo(modelInfo) {}

  LogicalResult
  matchAndRewrite(arc::SimStep op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    const ModelInfo *model = llvm::find_if(modelInfo, [&](auto x) {
      return x.name ==
             op.getInstance().getType().cast<SimModelInstanceType>().getModel();
    });
    if (model == modelInfo.end())
      return op.emitError("model not found");

    StringAttr evalFunc =
        rewriter.getStringAttr(evalSymbolFromModelName(model->name));
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, std::nullopt, evalFunc,
                                              adaptor.getInstance());

    return success();
  }

  ArrayRef<ModelInfo> modelInfo;
};

/// Lowers SimEmitLowering to a printf call. This pattern will mutate the global
/// module.
struct SimEmitLowering : public OpConversionPattern<arc::SimEmitValue> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arc::SimEmitValue op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto valueType = adaptor.getValue().getType().dyn_cast<IntegerType>();
    if (!valueType)
      return failure();

    // Surprisingly, pointer printing is the only somewhat portable way to print
    // in MLIR. This is because libc accepts arguments for which the size
    // depends on the ABI, which is not realistically accessible outside of
    // clang. The only part of the ABI that is predictable enough is pointer
    // size.
    llvm::TypeSize ptrSize = DataLayout::closest(op).getTypeSizeInBits(
        LLVM::LLVMPointerType::get(getContext()));

    uint64_t ptrSizeVal = ptrSize.getFixedValue();

    if (static_cast<uint64_t>(valueType.getWidth()) > ptrSizeVal)
      return op.emitError()
             << "printing integers of width " << valueType.getWidth()
             << " is not supported for this target triple (maximum is "
             << ptrSizeVal << ")";

    Location loc = op.getLoc();

    Value toPrint = rewriter.create<LLVM::IntToPtrOp>(
        loc, LLVM::LLVMPointerType::get(getContext()), adaptor.getValue());

    ModuleOp moduleOp = op->getParentOfType<ModuleOp>();
    if (!moduleOp)
      return failure();

    // Insert printf declaration if not available.
    auto printfType =
        LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(getContext()),
                                    LLVM::LLVMPointerType::get(getContext()),
                                    /*isVarArg=*/true);
    if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>("printf")) {
      ConversionPatternRewriter::InsertionGuard insertGuard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      rewriter.create<LLVM::LLVMFuncOp>(moduleOp.getLoc(), "printf",
                                        printfType);
    }

    FlatSymbolRefAttr printfSymbol =
        FlatSymbolRefAttr::get(getContext(), "printf");

    // Insert the format string if not already available.
    SmallString<16> formatStrName{"_arc_sim_emit_"};
    formatStrName.append(adaptor.getValueName());
    LLVM::GlobalOp formatStrGlobal;
    if (!(formatStrGlobal =
              moduleOp.lookupSymbol<LLVM::GlobalOp>(formatStrName))) {
      ConversionPatternRewriter::InsertionGuard insertGuard(rewriter);

      SmallString<16> formatStr = adaptor.getValueName();
      formatStr.append(" = %0.8p\n");
      SmallVector<char> formatStrVec{formatStr.begin(), formatStr.end()};
      formatStrVec.push_back(0);

      rewriter.setInsertionPointToStart(moduleOp.getBody());
      auto globalType =
          LLVM::LLVMArrayType::get(rewriter.getI8Type(), formatStrVec.size());
      formatStrGlobal = rewriter.create<LLVM::GlobalOp>(
          loc, globalType, /*isConstant=*/true, LLVM::Linkage::Internal,
          /*name=*/formatStrName, rewriter.getStringAttr(formatStrVec),
          /*alignment=*/0);
    }

    Value formatStrGlobalPtr =
        rewriter.create<LLVM::AddressOfOp>(loc, formatStrGlobal);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, printfType, printfSymbol, ValueRange{formatStrGlobalPtr, toPrint});

    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {
struct LowerArcToLLVMPass : public LowerArcToLLVMBase<LowerArcToLLVMPass> {
  void runOnOperation() override;
};
} // namespace

void LowerArcToLLVMPass::runOnOperation() {
  // Collect the symbols in the root op such that the HW-to-LLVM lowering can
  // create LLVM globals with non-colliding names.
  Namespace globals;
  SymbolCache cache;
  cache.addDefinitions(getOperation());
  globals.add(cache);

  // Setup the conversion target. Explicitly mark `scf.yield` legal since it
  // does not have a conversion itself, which would cause it to fail
  // legalization and for the conversion to abort. (It relies on its parent op's
  // conversion to remove it.)
  LLVMConversionTarget target(getContext());
  target.addLegalOp<mlir::ModuleOp>();
  target.addLegalOp<scf::YieldOp>(); // quirk of SCF dialect conversion

  // Setup the arc dialect type conversion.
  LLVMTypeConverter converter(&getContext());
  converter.addConversion([&](seq::ClockType type) {
    return IntegerType::get(type.getContext(), 1);
  });
  converter.addConversion([&](StorageType type) {
    return LLVM::LLVMPointerType::get(type.getContext());
  });
  converter.addConversion([&](MemoryType type) {
    return LLVM::LLVMPointerType::get(type.getContext());
  });
  converter.addConversion([&](StateType type) {
    return LLVM::LLVMPointerType::get(type.getContext());
  });
  converter.addConversion([&](SimModelInstanceType type) {
    return LLVM::LLVMPointerType::get(type.getContext());
  });

  // Setup the conversion patterns.
  RewritePatternSet patterns(&getContext());

  // MLIR patterns.
  populateSCFToControlFlowConversionPatterns(patterns);
  populateFuncToLLVMConversionPatterns(converter, patterns);
  cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
  arith::populateArithToLLVMConversionPatterns(converter, patterns);
  populateAnyFunctionOpInterfaceTypeConversionPattern(patterns, converter);

  // CIRCT patterns.
  DenseMap<std::pair<Type, ArrayAttr>, LLVM::GlobalOp> constAggregateGlobalsMap;
  populateHWToLLVMConversionPatterns(converter, patterns, globals,
                                     constAggregateGlobalsMap);
  populateHWToLLVMTypeConversions(converter);
  populateCombToLLVMConversionPatterns(converter, patterns);

  // Arc patterns.
  // clang-format off
  patterns.add<
    AllocMemoryOpLowering,
    AllocStateLikeOpLowering<arc::AllocStateOp>,
    AllocStateLikeOpLowering<arc::RootInputOp>,
    AllocStateLikeOpLowering<arc::RootOutputOp>,
    AllocStorageOpLowering,
    ClockGateOpLowering,
    MemoryReadOpLowering,
    MemoryWriteOpLowering,
    ModelOpLowering,
    ReplaceOpWithInputPattern<seq::ToClockOp>,
    ReplaceOpWithInputPattern<seq::FromClockOp>,
    SimEmitLowering,
    StateReadOpLowering,
    StateWriteOpLowering,
    StorageGetOpLowering,
    ZeroCountOpLowering
  >(converter, &getContext());
  // clang-format

  SmallVector<ModelInfo> models;
  if (failed(collectModels(getOperation(), models))) {
    signalPassFailure();
    return;
  }

  // clang-format off
  patterns.add<
    SimInstantiateLowering,
    SimSetInputLowering,
    SimGetPortLowering,
    SimStepLowering
  >(converter, &getContext(), models);
  // clang-format on

  // Apply the conversion.
  if (failed(applyFullConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> circt::createLowerArcToLLVMPass() {
  return std::make_unique<LowerArcToLLVMPass>();
}
