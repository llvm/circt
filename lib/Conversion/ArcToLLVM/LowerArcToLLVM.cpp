//===- LowerArcToLLVM.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/ArcToLLVM.h"
#include "circt/Conversion/CombToArith.h"
#include "circt/Conversion/CombToLLVM.h"
#include "circt/Conversion/HWToLLVM.h"
#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ModelInfo.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/ConversionPatternSet.h"
#include "circt/Support/Namespace.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "lower-arc-to-llvm"

namespace circt {
#define GEN_PASS_DEF_LOWERARCTOLLVM
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

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
      func::ReturnOp::create(rewriter, op.getLoc());
    }
    auto funcName =
        rewriter.getStringAttr(evalSymbolFromModelName(op.getName()));
    auto funcType =
        rewriter.getFunctionType(op.getBody().getArgumentTypes(), {});
    auto func =
        mlir::func::FuncOp::create(rewriter, op.getLoc(), funcName, funcType);
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
    Value ptr = LLVM::GEPOp::create(
        rewriter, op->getLoc(), adaptor.getStorage().getType(),
        rewriter.getI8Type(), adaptor.getStorage(),
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

struct AllocMemoryOpLowering : public OpConversionPattern<arc::AllocMemoryOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arc::AllocMemoryOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto offsetAttr = op->getAttrOfType<IntegerAttr>("offset");
    if (!offsetAttr)
      return failure();
    Value ptr = LLVM::GEPOp::create(
        rewriter, op.getLoc(), adaptor.getStorage().getType(),
        rewriter.getI8Type(), adaptor.getStorage(),
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
    Value offset = LLVM::ConstantOp::create(
        rewriter, op.getLoc(), rewriter.getI32Type(), op.getOffsetAttr());
    Value ptr = LLVM::GEPOp::create(
        rewriter, op.getLoc(), adaptor.getStorage().getType(),
        rewriter.getI8Type(), adaptor.getStorage(), offset);
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
      cast<IntegerType>(address.getType()).getWidth() + 1);
  Value addr = LLVM::ZExtOp::create(rewriter, loc, zextAddrType, address);
  Value addrLimit =
      LLVM::ConstantOp::create(rewriter, loc, zextAddrType,
                               rewriter.getI32IntegerAttr(type.getNumWords()));
  Value withinBounds = LLVM::ICmpOp::create(
      rewriter, loc, LLVM::ICmpPredicate::ult, addr, addrLimit);
  Value ptr = LLVM::GEPOp::create(
      rewriter, loc, LLVM::LLVMPointerType::get(memory.getContext()),
      rewriter.getIntegerType(type.getStride() * 8), memory, ValueRange{addr});
  return {ptr, withinBounds};
}

struct MemoryReadOpLowering : public OpConversionPattern<arc::MemoryReadOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arc::MemoryReadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto type = typeConverter->convertType(op.getType());
    auto memoryType = cast<MemoryType>(op.getMemory().getType());
    auto access =
        prepareMemoryAccess(op.getLoc(), adaptor.getMemory(),
                            adaptor.getAddress(), memoryType, rewriter);

    // Only attempt to read the memory if the address is within bounds,
    // otherwise produce a zero value.
    rewriter.replaceOpWithNewOp<scf::IfOp>(
        op, access.withinBounds,
        [&](auto &builder, auto loc) {
          Value loadOp = LLVM::LoadOp::create(
              builder, loc, memoryType.getWordType(), access.ptr);
          scf::YieldOp::create(builder, loc, loadOp);
        },
        [&](auto &builder, auto loc) {
          Value zeroValue = LLVM::ConstantOp::create(
              builder, loc, type, builder.getI64IntegerAttr(0));
          scf::YieldOp::create(builder, loc, zeroValue);
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
        cast<MemoryType>(op.getMemory().getType()), rewriter);
    auto enable = access.withinBounds;
    if (adaptor.getEnable())
      enable = LLVM::AndOp::create(rewriter, op.getLoc(), adaptor.getEnable(),
                                   enable);

    // Only attempt to write the memory if the address is within bounds.
    rewriter.replaceOpWithNewOp<scf::IfOp>(
        op, enable, [&](auto &builder, auto loc) {
          LLVM::StoreOp::create(builder, loc, adaptor.getData(), access.ptr);
          scf::YieldOp::create(builder, loc);
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
    rewriter.replaceOpWithNewOp<LLVM::AndOp>(op, adaptor.getInput(),
                                             adaptor.getEnable());
    return success();
  }
};

/// Lower 'seq.clock_inv x' to 'llvm.xor x true'
struct ClockInvOpLowering : public OpConversionPattern<seq::ClockInverterOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(seq::ClockInverterOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto constTrue = LLVM::ConstantOp::create(rewriter, op->getLoc(),
                                              rewriter.getI1Type(), 1);
    rewriter.replaceOpWithNewOp<LLVM::XOrOp>(op, adaptor.getInput(), constTrue);
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

struct SeqConstClockLowering : public OpConversionPattern<seq::ConstClockOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(seq::ConstClockOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
        op, rewriter.getI1Type(), static_cast<int64_t>(op.getValue()));
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

namespace {

// Non-model specific types and symbols.
struct GlobalTraceHelpers {

  /*
  struct ArcTraceModelInfo {
    uint64_t numTraceTaps;
    char* modelName;
    char* signalNames;
    uint64_t* tapNameOffsets;
    uint64_t* typeDescriptors;
    uint64_t* stateOffsets;
  };
  */
  LLVM::LLVMStructType modelInfoStructType;
  static constexpr int modelInfoStructNumTraceTapsField = 0;
  static constexpr int modelInfoStructModelNameField = 1;
  static constexpr int modelInfoStructSignalNamesField = 2;
  static constexpr int modelInfoStructTapNameOffsetsField = 3;
  static constexpr int modelInfoStructTypeDescriptors = 4;
  static constexpr int modelInfoStructStateOffsets = 5;

  /*
    struct ArcTracerState {
      void* buffer;
      uint64_t size;
      uint64_t capacity;
      uint64_t runSteps;
      void* user;
    };
  */
  LLVM::LLVMStructType tracerStateStructType;
  static constexpr int tracerStateStructBufferField = 0;
  static constexpr int tracerStateStructSizeField = 1;
  static constexpr int tracerStateStructCapacityField = 2;
  static constexpr int tracerStateStructRunStepsField = 3;
  static constexpr int tracerStateStructUserField = 4;

  static constexpr int tracerStateSize = 5 * 8;

  /*
    struct ArcTraceLibrary {
      void* (*initModel)(ArcTraceModelInfo* modelInfo);
      void  (*step)(void* state);
      void* (*swapBuffer)(void* oldBuffer, uint64_t bufferSize, void* user);
      void  (*closeModel)(void* state)
    };
  */
  LLVM::LLVMStructType libraryStructType;
  static constexpr int libraryStructInitModelField = 0;
  static constexpr int libraryStructStepField = 1;
  static constexpr int libraryStructSwapBufferField = 2;
  static constexpr int libraryStructCloseModelField = 3;

  LLVM::LLVMFunctionType initModelFnTy;
  LLVM::LLVMFunctionType stepFnTy;
  LLVM::LLVMFunctionType swapBufferFnType;
  LLVM::LLVMFunctionType closeModelFnTy;

  DictionaryAttr noFreeNoCapParamAttr;
  DictionaryAttr noAliasAttr;

  explicit GlobalTraceHelpers(MLIRContext *context) {
    auto i64Ty = IntegerType::get(context, 64);
    auto ptrTy = LLVM::LLVMPointerType::get(context);
    auto voidTy = LLVM::LLVMVoidType::get(context);

    modelInfoStructType = LLVM::LLVMStructType::getLiteral(
        context, {i64Ty, ptrTy, ptrTy, ptrTy, ptrTy, ptrTy});

    tracerStateStructType = LLVM::LLVMStructType::getLiteral(
        context, {ptrTy, i64Ty, i64Ty, i64Ty, ptrTy});

    libraryStructType =
        LLVM::LLVMStructType::getLiteral(context, {ptrTy, ptrTy, ptrTy, ptrTy});

    initModelFnTy = LLVM::LLVMFunctionType::get(ptrTy, {ptrTy});
    stepFnTy = LLVM::LLVMFunctionType::get(voidTy, {ptrTy});
    swapBufferFnType = LLVM::LLVMFunctionType::get(ptrTy, {ptrTy});
    closeModelFnTy = LLVM::LLVMFunctionType::get(ptrTy, {ptrTy});

    NamedAttrList fields;
    fields.append(LLVM::LLVMDialect::getNoAliasAttrName(),
                  UnitAttr::get(context));
    noAliasAttr = fields.getDictionary(context);

    fields.clear();
    fields.append(LLVM::LLVMDialect::getNoCaptureAttrName(),
                  UnitAttr::get(context));
    fields.append(LLVM::LLVMDialect::getNoFreeAttrName(),
                  UnitAttr::get(context));

    noFreeNoCapParamAttr = fields.getDictionary(context);
  }

  LLVM::LLVMFuncOp registerLibraryFn;
  LLVM::GlobalOp libraryStruct;
  llvm::SmallDenseMap<unsigned, LLVM::LLVMFuncOp> bufferAppendFns;

  static unsigned getCapacityForBitWidth(unsigned bitwidth) {
    assert(bitwidth != 0);
    return (bitwidth % 64 == 0) ? bitwidth / 64 : bitwidth / 64 + 1;
  }

  LogicalResult buildAppendFunctions(ImplicitLocOpBuilder builder,
                                     ArrayRef<ModelOp> traceableModels) {

    auto ptrTy = LLVM::LLVMPointerType::get(builder.getContext());
    auto i64Ty = builder.getI64Type();
    for (auto model : traceableModels) {
      for (auto tap : model.getTraceTapsAttr()) {
        auto tapAttr = cast<TraceTapAttr>(tap);
        auto tapType = dyn_cast<IntegerType>(tapAttr.getSigType().getValue());
        if (!tapType) {
          model.emitError()
              << "Trace Tap of non-integer type "
              << tapAttr.getSigType().getValue() << " unsupported.";
          return failure();
        }
        auto numWords = getCapacityForBitWidth(tapType.getIntOrFloatBitWidth());
        if (bufferAppendFns.contains(numWords))
          continue;
        OpBuilder::InsertionGuard g(builder);
        auto fnName = builder.getStringAttr("_arc_trace_tap_append_i" +
                                            Twine(numWords * 64));
        auto extType = builder.getIntegerType(numWords * 64);
        // void _arc_trace_tap_append_ix(uint8_t* state, uint64_t tapId, ix
        // newValue)
        auto fnType = LLVM::LLVMFunctionType::get(
            LLVM::LLVMVoidType::get(builder.getContext()),
            {ptrTy, i64Ty, extType});
        auto appendFn = LLVM::LLVMFuncOp::create(builder, fnName, fnType,
                                                 LLVM::Linkage::Private);
        bufferAppendFns.insert({numWords, appendFn});

        auto *hasTracerCheckBlock = appendFn.addEntryBlock(builder);
        auto *capcaityCheckBlock = &appendFn.getRegion().emplaceBlock();
        auto *swapBufferBlock = &appendFn.getRegion().emplaceBlock();
        auto *bufferStoreBlock = &appendFn.getRegion().emplaceBlock();
        bufferStoreBlock->addArgument(ptrTy, builder.getLoc());
        bufferStoreBlock->addArgument(i64Ty, builder.getLoc());
        auto *exitBlock = &appendFn.getRegion().emplaceBlock();

        // Check if we actually have a trace buffer
        builder.setInsertionPointToStart(hasTracerCheckBlock);
        auto traceStatePtr = hasTracerCheckBlock->getArgument(0);
        auto bufferPtrPtr = LLVM::GEPOp::create(
            builder, ptrTy, tracerStateStructType, traceStatePtr,
            {LLVM::GEPArg(0),
             LLVM::GEPArg(GlobalTraceHelpers::tracerStateStructBufferField)});
        auto bufferPtrVal = LLVM::LoadOp::create(builder, ptrTy, bufferPtrPtr);
        auto nullPtrVal = LLVM::ZeroOp::create(builder, ptrTy);
        auto bufferIsNull = LLVM::ICmpOp::create(
            builder, LLVM::ICmpPredicate::eq, bufferPtrVal, nullPtrVal);
        LLVM::CondBrOp::create(builder, bufferIsNull, exitBlock, {},
                               capcaityCheckBlock, {});

        // We've got one. Check if we are running out of space.
        builder.setInsertionPointToStart(capcaityCheckBlock);
        auto sizePtr = LLVM::GEPOp::create(
            builder, ptrTy, tracerStateStructType, traceStatePtr,
            {LLVM::GEPArg(0),
             LLVM::GEPArg(GlobalTraceHelpers::tracerStateStructSizeField)});
        auto capacityPtr = LLVM::GEPOp::create(
            builder, ptrTy, tracerStateStructType, traceStatePtr,
            {LLVM::GEPArg(0),
             LLVM::GEPArg(GlobalTraceHelpers::tracerStateStructCapacityField)});
        auto sizeVal = LLVM::LoadOp::create(builder, i64Ty, sizePtr);
        auto capacityVal = LLVM::LoadOp::create(builder, i64Ty, capacityPtr);

        auto requiredCapacity = numWords + 1;
        auto resizeCst = LLVM::ConstantOp::create(
            builder, builder.getI64IntegerAttr(requiredCapacity));
        auto newSizeVal = LLVM::AddOp::create(builder, sizeVal, resizeCst);
        auto storeOffset = LLVM::GEPOp::create(
            builder, ptrTy, i64Ty, bufferPtrVal, {LLVM::GEPArg(sizeVal)});
        auto needsSwap = LLVM::ICmpOp::create(builder, LLVM::ICmpPredicate::ugt,
                                              newSizeVal, capacityVal);
        LLVM::CondBrOp::create(builder, needsSwap, swapBufferBlock, {},
                               bufferStoreBlock, {storeOffset, newSizeVal},
                               std::pair<int32_t, int32_t>(
                                   0, std::numeric_limits<int32_t>::max()));

        // Too bad, we need a new buffer. Call the library.
        builder.setInsertionPointToStart(swapBufferBlock);
        auto userPtrPtr = LLVM::GEPOp::create(
            builder, ptrTy, tracerStateStructType, traceStatePtr,
            {LLVM::GEPArg(0),
             LLVM::GEPArg(GlobalTraceHelpers::tracerStateStructUserField)});
        auto userPtrVal = LLVM::LoadOp::create(builder, ptrTy, userPtrPtr);
        auto libraryStuctPtr =
            LLVM::AddressOfOp::create(builder, libraryStruct);
        auto swapBufferFnPtr = LLVM::GEPOp::create(
            builder, ptrTy, libraryStructType, libraryStuctPtr,
            {LLVM::GEPArg(0),
             LLVM::GEPArg(GlobalTraceHelpers::libraryStructSwapBufferField)});
        auto swapCall = LLVM::CallOp::create(
            builder, swapBufferFnType,
            {swapBufferFnPtr, bufferPtrVal, sizeVal, userPtrVal});
        swapCall.setResAttrsAttr(builder.getArrayAttr({noAliasAttr}));
        LLVM::StoreOp::create(builder, swapCall.getResult(), bufferPtrPtr);
        LLVM::BrOp::create(builder, {swapCall.getResult(), resizeCst},
                           bufferStoreBlock);

        // Store the tap index and new value in the buffer and update the size.
        builder.setInsertionPointToStart(bufferStoreBlock);
        auto dataStorePtr = LLVM::GEPOp::create(
            builder, ptrTy, i64Ty, bufferStoreBlock->getArgument(0),
            {LLVM::GEPArg(1)});
        LLVM::StoreOp::create(builder, hasTracerCheckBlock->getArgument(1),
                              bufferStoreBlock->getArgument(0));
        LLVM::StoreOp::create(builder, hasTracerCheckBlock->getArgument(2),
                              dataStorePtr);
        LLVM::StoreOp::create(builder, bufferStoreBlock->getArgument(1),
                              sizePtr);
        LLVM::BrOp::create(builder, exitBlock);

        // And we're done
        builder.setInsertionPointToStart(exitBlock);
        LLVM::ReturnOp::create(builder, Value{});
      }
    }
    return success();
  }

  void buildModuleGlobals(ImplicitLocOpBuilder builder,
                          unsigned numTraceableModels) {
    auto ptrTy = LLVM::LLVMPointerType::get(builder.getContext());
    libraryStruct =
        LLVM::GlobalOp::create(builder, libraryStructType,
                               /*isConstant*/ false, LLVM::Linkage::Internal,
                               "_arc_trace_library", Attribute{});
    {
      OpBuilder::InsertionGuard g(builder);
      Region &initRegion = libraryStruct.getInitializerRegion();
      Block *initBlock = builder.createBlock(&initRegion);
      builder.setInsertionPointToStart(initBlock);
      auto undefStruct = LLVM::UndefOp::create(builder, libraryStructType);
      Value currentStruct = undefStruct;
      auto nullPtr = LLVM::ZeroOp::create(builder, ptrTy);
      for (int64_t i = 0; i < libraryStructCloseModelField + 1; ++i)
        currentStruct =
            LLVM::InsertValueOp::create(builder, currentStruct, nullPtr, {i});
      LLVM::ReturnOp::create(builder, currentStruct);
    }
    auto fnTy = LLVM::LLVMFunctionType::get(builder.getI32Type(), {ptrTy});
    auto fnName = builder.getStringAttr(Twine("_mlir_ciface_") +
                                        globalRegisterTraceLibrarySymName);
    registerLibraryFn = LLVM::LLVMFuncOp::create(builder, fnName, fnTy,
                                                 LLVM::Linkage::External);
    {
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointToStart(
          registerLibraryFn.addEntryBlock(builder));
      auto globAddr = LLVM::AddressOfOp::create(builder, libraryStruct);
      auto load = LLVM::LoadOp::create(builder, libraryStructType,
                                       builder.getBlock()->getArgument(0));
      LLVM::StoreOp::create(builder, load.getResult(), globAddr);
      auto numModelsCst = LLVM::ConstantOp::create(
          builder, builder.getI32IntegerAttr(numTraceableModels));
      LLVM::ReturnOp::create(builder, numModelsCst);
    }
  }
};

// Model specific tracing logic
struct ModelTraceHelpers {
public:
  ModelTraceHelpers(GlobalTraceHelpers &globals, ModelOp modelOp)
      : modelName(modelOp.getNameAttr()), traceTaps(modelOp.getTraceTapsAttr()),
        modLoc(modelOp.getLoc()), globals(globals) {}

  // Call the library's `step` function. This should be done _before_ every
  // model evaluation.
  Operation *buildStepCall(Value baseState, Location loc,
                           ConversionPatternRewriter &rewriter) {
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto tracerLibAddr =
        LLVM::AddressOfOp::create(rewriter, loc, globals.libraryStruct);

    auto stepFnPtr = LLVM::GEPOp::create(
        rewriter, loc, ptrTy, globals.libraryStructType, tracerLibAddr,
        {LLVM::GEPArg(0),
         LLVM::GEPArg(GlobalTraceHelpers::libraryStructStepField)});
    auto stepFnVal = LLVM::LoadOp::create(rewriter, loc, ptrTy, stepFnPtr);
    auto nullPtr = LLVM::ZeroOp::create(rewriter, loc, ptrTy);
    auto fnNotNull = LLVM::ICmpOp::create(
        rewriter, loc, LLVM::ICmpPredicate::ne, stepFnVal, nullPtr);

    auto scfIf = scf::IfOp::create(
        rewriter, loc, fnNotNull, [&](OpBuilder &builder, Location loc) {
          auto traceStatePtr = LLVM::GEPOp::create(
              rewriter, loc, ptrTy, rewriter.getI8Type(), baseState,
              {LLVM::GEPArg(-1 * static_cast<int32_t>(
                                     GlobalTraceHelpers::tracerStateSize))});
          auto callOp = LLVM::CallOp::create(builder, loc, globals.stepFnTy,
                                             {stepFnVal, traceStatePtr});
          callOp.setArgAttrsAttr(
              rewriter.getArrayAttr({globals.noFreeNoCapParamAttr}));
          scf::YieldOp::create(builder, loc);
        });
    return scfIf;
  }

  void lowerTappedStateWrite(arc::StateWriteOp op,
                             arc::StateWriteOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {

    assert(op.getTraceTapModelAttr() == modelName);
    assert(op.getTraceTapIndex().has_value());
    assert(isa<IntegerType>(adaptor.getValue().getType()) && "Unsupported");

    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    uint64_t baseOffset =
        cast<TraceTapAttr>(traceTaps[op.getTraceTapIndex()->getZExtValue()])
            .getStateOffset() +
        GlobalTraceHelpers::tracerStateSize;
    assert(baseOffset <= std::numeric_limits<int32_t>::max());

    // Check if the state value has changed. If not, bail.
    auto oldVal =
        LLVM::LoadOp::create(rewriter, op.getLoc(),
                             adaptor.getValue().getType(), adaptor.getState());
    auto hasChanged =
        LLVM::ICmpOp::create(rewriter, op.getLoc(), LLVM::ICmpPredicate::ne,
                             adaptor.getValue(), oldVal);
    scf::IfOp::create(
        rewriter, op.getLoc(), hasChanged,
        [&](OpBuilder builder, Location loc) {
          // Call the function to append the trace buffer.
          auto typeBits = cast<IntegerType>(adaptor.getValue().getType())
                              .getIntOrFloatBitWidth();
          auto numWords = GlobalTraceHelpers::getCapacityForBitWidth(
              cast<IntegerType>(adaptor.getValue().getType())
                  .getIntOrFloatBitWidth());
          auto appendFn = globals.bufferAppendFns.at(numWords);

          auto traceStatePtr = LLVM::GEPOp::create(
              builder, loc, ptrTy, rewriter.getI8Type(), adaptor.getState(),
              {LLVM::GEPArg(-1 * static_cast<int32_t>(baseOffset))});

          auto tapIdxCst = LLVM::ConstantOp::create(
              builder, loc,
              rewriter.getI64IntegerAttr(
                  op.getTraceTapIndexAttr().getValue().getZExtValue()));

          auto storeVal = adaptor.getValue();
          if (typeBits != numWords * 64)
            storeVal = LLVM::ZExtOp::create(
                           rewriter, op.getLoc(),
                           rewriter.getIntegerType(numWords * 64), storeVal)
                           .getResult();
          LLVM::CallOp::create(builder, loc, appendFn,
                               {traceStatePtr, tapIdxCst, storeVal});
          scf::YieldOp::create(builder, loc);
        });
    // Store the value to the state
    LLVM::StoreOp::create(rewriter, op.getLoc(), adaptor.getValue(),
                          adaptor.getState());
  }

  GlobalTraceHelpers &getGlobals() const { return globals; }

  // Create all the model specific constants the tracing library needs.
  void buildStaticModuleStructs(OpBuilder &argBuilder) {
    ImplicitLocOpBuilder builder(modLoc, argBuilder);
    buildSignalNamesArrays(builder);
    buildTypeDescriptorsArray(builder);
    buildValueOffsetsArray(builder);

    StringAttr modelNameSym =
        builder.getStringAttr("_arc_model_trace_name_" + Twine(modelName));
    StringAttr modelTraceInfoStructSymName =
        builder.getStringAttr("_arc_model_trace_info_" + Twine(modelName));

    SmallVector<char, 16> nameBuffer;
    nameBuffer.append(modelName.begin(), modelName.end());
    nameBuffer.push_back('\0');

    auto charArrayType =
        LLVM::LLVMArrayType::get(builder.getI8Type(), nameBuffer.size());
    auto modNameStr = LLVM::GlobalOp::create(
        builder, charArrayType,
        /*isConstant=*/true, LLVM::Linkage::Internal,
        /*name=*/modelNameSym, builder.getStringAttr(nameBuffer),
        /*alignment=*/0);

    traceInfoStruct =
        LLVM::GlobalOp::create(builder, globals.modelInfoStructType,
                               /*isConstant=*/true, LLVM::Linkage::Internal,
                               modelTraceInfoStructSymName, Attribute{});

    /*
      struct ArcTraceModelInfo {
        uint64_t numTraceTaps;
        char *modelName;
        char *signalNames;
        uint64_t *tapNameOffsets;
        uint64_t *typeDescriptors;
        uint64_t *stateOffsets;
      };
    */

    // Create the initializer region
    Region &initRegion = traceInfoStruct.getInitializerRegion();
    Block *initBlock = builder.createBlock(&initRegion);
    builder.setInsertionPointToStart(initBlock);

    auto undefStruct =
        LLVM::UndefOp::create(builder, globals.modelInfoStructType);
    Value currentStruct = undefStruct;

    // Field 0: numTraceTaps
    auto numSigsCst = LLVM::ConstantOp::create(
        builder, builder.getI64IntegerAttr(traceTaps.size()));
    currentStruct = LLVM::InsertValueOp::create(
        builder, currentStruct, numSigsCst,
        ArrayRef<int64_t>{
            GlobalTraceHelpers::modelInfoStructNumTraceTapsField});
    // Field 1: modelName
    auto namePtr = LLVM::AddressOfOp::create(builder, modNameStr);
    currentStruct = LLVM::InsertValueOp::create(
        builder, currentStruct, namePtr,
        ArrayRef<int64_t>{GlobalTraceHelpers::modelInfoStructModelNameField});
    // Field 2: signalNames
    auto sigNamesPtr = LLVM::AddressOfOp::create(builder, signalNamesArray);
    currentStruct = LLVM::InsertValueOp::create(
        builder, currentStruct, sigNamesPtr,
        ArrayRef<int64_t>{GlobalTraceHelpers::modelInfoStructSignalNamesField});

    // Field 3: tapNameOffsets
    auto nameOffsetsPtr =
        LLVM::AddressOfOp::create(builder, signalNameOffsetsArray);
    currentStruct = LLVM::InsertValueOp::create(
        builder, currentStruct, nameOffsetsPtr,
        ArrayRef<int64_t>{
            GlobalTraceHelpers::modelInfoStructTapNameOffsetsField});

    // Field 4: typeDescriptors
    auto typeDescsPtr =
        LLVM::AddressOfOp::create(builder, typeDescriptorsArray);
    currentStruct = LLVM::InsertValueOp::create(
        builder, currentStruct, typeDescsPtr,
        ArrayRef<int64_t>{GlobalTraceHelpers::modelInfoStructTypeDescriptors});

    // Field 5: stateOffsets
    auto offsetsPtr = LLVM::AddressOfOp::create(builder, valueOffsetsArray);
    currentStruct = LLVM::InsertValueOp::create(
        builder, currentStruct, offsetsPtr,
        ArrayRef<int64_t>{GlobalTraceHelpers::modelInfoStructStateOffsets});

    LLVM::ReturnOp::create(builder, currentStruct);
  }

  LLVM::GlobalOp traceInfoStruct;

private:
  // Create the list of names/aliases and their respective offsets.
  void buildSignalNamesArrays(ImplicitLocOpBuilder &builder) {
    SmallVector<char> sigNames;
    SmallVector<int64_t> sigNameOffsets;

    auto sigNamesSym =
        builder.getStringAttr("_arc_tracer_signal_names_" + Twine(modelName));
    auto sigNameOffsetsSym = builder.getStringAttr(
        "_arc_tracer_signal_name_offsets_" + Twine(modelName));

    sigNames.reserve(traceTaps.size() * 32);
    sigNameOffsets.reserve(traceTaps.size());
    for (auto tap : traceTaps) {
      for (auto name : cast<TraceTapAttr>(tap).getNames()) {
        auto nameStrAttr = cast<StringAttr>(name);
        sigNames.append(nameStrAttr.begin(), nameStrAttr.end());
        sigNames.push_back('\0');
      }
      sigNameOffsets.push_back(sigNames.size());
    }

    auto charArrayType =
        LLVM::LLVMArrayType::get(builder.getI8Type(), sigNames.size());
    signalNamesArray = LLVM::GlobalOp::create(
        builder, charArrayType,
        /*isConstant=*/true, LLVM::Linkage::Internal,
        /*name=*/sigNamesSym, builder.getStringAttr(sigNames),
        /*alignment=*/0);

    auto arrayTy =
        LLVM::LLVMArrayType::get(builder.getI64Type(), sigNameOffsets.size());
    auto tensorType = RankedTensorType::get(
        {static_cast<int64_t>(sigNameOffsets.size())}, builder.getI64Type());
    auto denseAttr = DenseIntElementsAttr::get(tensorType, sigNameOffsets);

    signalNameOffsetsArray =
        LLVM::GlobalOp::create(builder, arrayTy,
                               /*isConstant=*/true, LLVM::Linkage::Internal,
                               /*name=*/sigNameOffsetsSym, denseAttr,
                               /*alignment=*/0);
  }

  // Create the array of per-tap type descriptors
  void buildTypeDescriptorsArray(ImplicitLocOpBuilder &builder) {
    auto typeDescriptorsSym = builder.getStringAttr(
        "_arc_tracer_type_descriptors_" + Twine(modelName));
    SmallVector<int32_t> rawTypeDescs;
    rawTypeDescs.reserve(traceTaps.size());
    for (auto tap : traceTaps) {
      auto tapAttr = cast<TraceTapAttr>(tap);
      assert(isa<IntegerType>(tapAttr.getSigType().getValue()) &&
             "Unsupported");
      rawTypeDescs.push_back(static_cast<int32_t>(
          tapAttr.getSigType().getValue().getIntOrFloatBitWidth()));
    }

    auto arrayTy =
        LLVM::LLVMArrayType::get(builder.getI32Type(), rawTypeDescs.size());
    auto tensorType = RankedTensorType::get(
        {static_cast<int64_t>(rawTypeDescs.size())}, builder.getI32Type());
    auto denseAttr = DenseIntElementsAttr::get(tensorType, rawTypeDescs);

    typeDescriptorsArray =
        LLVM::GlobalOp::create(builder, arrayTy,
                               /*isConstant=*/true, LLVM::Linkage::Internal,
                               /*name=*/typeDescriptorsSym, denseAttr,
                               /*alignment=*/0);
  }

  // Create the array of per-tap value offsets in the simulation state.
  // We need them to be able to do a full dump of all taps.
  void buildValueOffsetsArray(ImplicitLocOpBuilder &builder) {
    auto valueOffsetsSym =
        builder.getStringAttr("_arc_tracer_value_offsets_" + Twine(modelName));
    SmallVector<uint64_t> offsets;
    offsets.reserve(traceTaps.size());
    for (auto tap : traceTaps)
      offsets.push_back(cast<TraceTapAttr>(tap).getStateOffset());
    auto arrayTy =
        LLVM::LLVMArrayType::get(builder.getI64Type(), offsets.size());
    auto tensorType = RankedTensorType::get(
        {static_cast<int64_t>(offsets.size())}, builder.getI64Type());
    auto denseAttr = DenseIntElementsAttr::get(tensorType, offsets);
    valueOffsetsArray =
        LLVM::GlobalOp::create(builder, arrayTy,
                               /*isConstant=*/true, LLVM::Linkage::Internal,
                               /*name=*/valueOffsetsSym, denseAttr,
                               /*alignment=*/0);
  }

  StringAttr modelName;
  ArrayAttr  traceTaps;
  Location modLoc;
  GlobalTraceHelpers &globals;

  LLVM::GlobalOp signalNamesArray;
  LLVM::GlobalOp signalNameOffsetsArray;
  LLVM::GlobalOp typeDescriptorsArray;
  LLVM::GlobalOp valueOffsetsArray;
};


struct StateWriteOpLowering : public OpConversionPattern<arc::StateWriteOp> {
  using OpConversionPattern::OpConversionPattern;

  StateWriteOpLowering(
      const TypeConverter &typeConverter, MLIRContext *context,
      llvm::SmallDenseMap<StringRef, ModelTraceHelpers> &tracerMap)
      : OpConversionPattern<arc::StateWriteOp>(typeConverter, context),
        tracerMap(tracerMap) {}

  LogicalResult
  matchAndRewrite(arc::StateWriteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    bool hasTracer = op.getTraceTapModel().has_value();

    // If the state write is tapped, the logic gets a _bit_ more complicated.
    auto lowerTapped = [&](){
      auto traceIt = tracerMap.find(op.getTraceTapModelAttr());
      assert(traceIt != tracerMap.end());
      traceIt->second.lowerTappedStateWrite(op, adaptor, rewriter);
    };

    if (adaptor.getCondition()) {
      scf::IfOp::create(rewriter, op.getLoc(), adaptor.getCondition(), [&](auto &builder, auto loc) {
            if (!hasTracer)
              LLVM::StoreOp::create(builder, loc, adaptor.getValue(),
                                    adaptor.getState());
            else
              lowerTapped();
            scf::YieldOp::create(builder, loc);
          });
      rewriter.eraseOp(op);
    } else {
      if (!hasTracer) {
        rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, adaptor.getValue(),
                                                  adaptor.getState());
      } else {
        lowerTapped();
        rewriter.eraseOp(op);
      }
    }
    return success();
  }

  llvm::SmallDenseMap<StringRef, ModelTraceHelpers> &tracerMap;
};

}


//===----------------------------------------------------------------------===//
// Simulation Orchestration Lowering Patterns
//===----------------------------------------------------------------------===//

namespace {

struct ModelInfoMap {
  size_t numStateBytes;
  llvm::DenseMap<StringRef, StateInfo> states;
  mlir::FlatSymbolRefAttr initialFnSymbol;
  mlir::FlatSymbolRefAttr finalFnSymbol;
};

template <typename OpTy>
struct ModelAwarePattern : public OpConversionPattern<OpTy> {
  ModelAwarePattern(const TypeConverter &typeConverter, MLIRContext *context,
                    llvm::DenseMap<StringRef, ModelInfoMap> &modelInfo, llvm::SmallDenseMap<StringRef, ModelTraceHelpers>& tracerMap)
      : OpConversionPattern<OpTy>(typeConverter, context),
        modelInfo(modelInfo), tracerMap(tracerMap) {}

protected:
  Value createPtrToPortState(ConversionPatternRewriter &rewriter, Location loc,
                             Value state, const StateInfo &port) const {
    MLIRContext *ctx = rewriter.getContext();
    return LLVM::GEPOp::create(rewriter, loc, LLVM::LLVMPointerType::get(ctx),
                               IntegerType::get(ctx, 8), state,
                               LLVM::GEPArg(port.offset));
  }

  llvm::DenseMap<StringRef, ModelInfoMap> &modelInfo;
  llvm::SmallDenseMap<StringRef, ModelTraceHelpers> &tracerMap;
};

/// Lowers SimInstantiateOp to a malloc and memset call. This pattern will
/// mutate the global module.
struct SimInstantiateOpLowering
    : public ModelAwarePattern<arc::SimInstantiateOp> {
  using ModelAwarePattern::ModelAwarePattern;

  LogicalResult
  matchAndRewrite(arc::SimInstantiateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto modelName = cast<SimModelInstanceType>(op.getBody().getArgument(0).getType())
            .getModel()
            .getValue();

    auto modelIt = modelInfo.find(modelName);
    auto tracer = tracerMap.find(modelName);
    bool hasTracer = tracer != tracerMap.end();
    ModelInfoMap &model = modelIt->second;


    ModuleOp moduleOp = op->getParentOfType<ModuleOp>();
    if (!moduleOp)
      return failure();

    ConversionPatternRewriter::InsertionGuard guard(rewriter);

    // FIXME: like the rest of MLIR, this assumes sizeof(intptr_t) ==
    // sizeof(size_t) on the target architecture.
    Type convertedIndex = typeConverter->convertType(rewriter.getIndexType());

    FailureOr<LLVM::LLVMFuncOp> mallocFunc =
        LLVM::lookupOrCreateMallocFn(rewriter, moduleOp, convertedIndex);
    if (failed(mallocFunc))
      return mallocFunc;

    FailureOr<LLVM::LLVMFuncOp> freeFunc =
        LLVM::lookupOrCreateFreeFn(rewriter, moduleOp);
    if (failed(freeFunc))
      return freeFunc;

    size_t numStateBytes = model.numStateBytes;
    // If we implement tracing, we need to allocate extra space for
    // the dynamic trace info.
    if (hasTracer)
      numStateBytes += GlobalTraceHelpers::tracerStateSize;

    auto ptrTy = LLVM::LLVMPointerType::get(op.getContext());
    Location loc = op.getLoc();
    Value numStateBytesCst = LLVM::ConstantOp::create(
        rewriter, loc, convertedIndex, numStateBytes);
    Value allocated = LLVM::CallOp::create(rewriter, loc, mallocFunc.value(),
                                           ValueRange{numStateBytesCst})
                          .getResult();

    Value zero =
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI8Type(), 0);
    LLVM::MemsetOp::create(rewriter, loc, allocated, zero, numStateBytesCst,
                           false);

    Value simState = allocated;
    if (hasTracer) {
      // Move the 'actual' state pointer beyond the tracer state
      simState = LLVM::GEPOp::create(
          rewriter, loc, ptrTy, rewriter.getI8Type(), allocated,
          LLVM::GEPArg(GlobalTraceHelpers::tracerStateSize));
      // Call the tracer libraries `initModel` function, if we've got one.
      auto tracerLibAddr = LLVM::AddressOfOp::create(
          rewriter, loc, tracer->second.getGlobals().libraryStruct);
      auto initFnPtr = LLVM::GEPOp::create(
          rewriter, loc, ptrTy, tracer->second.getGlobals().libraryStructType,
          tracerLibAddr,
          {LLVM::GEPArg(0),
           LLVM::GEPArg(GlobalTraceHelpers::libraryStructInitModelField)});
      auto initFnVal = LLVM::LoadOp::create(rewriter, loc, ptrTy, initFnPtr);
      auto nullPtr = LLVM::ZeroOp::create(rewriter, loc, ptrTy);
      auto fnNotNull = LLVM::ICmpOp::create(
          rewriter, loc, LLVM::ICmpPredicate::ne, initFnVal, nullPtr);
      scf::IfOp::create(
          rewriter, loc, fnNotNull, [&](OpBuilder &builder, Location loc) {
            auto modelInfoAddr = LLVM::AddressOfOp::create(
                builder, loc, tracer->second.traceInfoStruct);
            auto libInitCall = LLVM::CallOp::create(
                builder, loc, tracer->second.getGlobals().initModelFnTy,
                {initFnVal, modelInfoAddr});
            auto userPtrPtr = LLVM::GEPOp::create(
                builder, loc, ptrTy,
                tracer->second.getGlobals().tracerStateStructType, allocated,
                {LLVM::GEPArg(0),
                 LLVM::GEPArg(GlobalTraceHelpers::tracerStateStructUserField)});
            LLVM::StoreOp::create(builder, loc, libInitCall.getResult(),
                                  userPtrPtr.getResult());
            scf::YieldOp::create(builder, loc);
          });
    }

    // Call the model's 'initial' function if present.
    if (model.initialFnSymbol) {
      if (hasTracer)
        tracer->second.buildStepCall(simState, op.getLoc(), rewriter);

      auto initialFnType = LLVM::LLVMFunctionType::get(
          LLVM::LLVMVoidType::get(op.getContext()), {ptrTy});
      LLVM::CallOp::create(rewriter, loc, initialFnType, model.initialFnSymbol,
                           ValueRange{simState});
    }

    // Execute the body.
    rewriter.inlineBlockBefore(&adaptor.getBody().getBlocks().front(), op,
                               {simState});

    // Call the model's 'final' function if present.
    if (model.finalFnSymbol) {
      if (hasTracer)
        tracer->second.buildStepCall(simState, op.getLoc(), rewriter);
      auto finalFnType = LLVM::LLVMFunctionType::get(
          LLVM::LLVMVoidType::get(op.getContext()),
          {LLVM::LLVMPointerType::get(op.getContext())});
      LLVM::CallOp::create(rewriter, loc, finalFnType, model.finalFnSymbol,
                           ValueRange{simState});
    }

    if (hasTracer) {
      // Call the tracer libraries `closeModel` function, if we've got one.
      auto tracerLibAddr = LLVM::AddressOfOp::create(
          rewriter, loc, tracer->second.getGlobals().libraryStruct);
      auto closeFnPtr = LLVM::GEPOp::create(
          rewriter, loc, ptrTy, tracer->second.getGlobals().libraryStructType,
          tracerLibAddr,
          {LLVM::GEPArg(0),
           LLVM::GEPArg(GlobalTraceHelpers::libraryStructCloseModelField)});
      auto closeFnVal = LLVM::LoadOp::create(rewriter, loc, ptrTy, closeFnPtr);
      auto nullPtr = LLVM::ZeroOp::create(rewriter, loc, ptrTy);
      auto fnNotNull = LLVM::ICmpOp::create(
          rewriter, loc, LLVM::ICmpPredicate::ne, closeFnVal, nullPtr);
      scf::IfOp::create(
          rewriter, loc, fnNotNull, [&](OpBuilder &builder, Location loc) {
            LLVM::CallOp::create(builder, loc,
                                 tracer->second.getGlobals().initModelFnTy,
                                 {closeFnVal, allocated});
            scf::YieldOp::create(builder, loc);
          });
    }

    LLVM::CallOp::create(rewriter, loc, freeFunc.value(),
                         ValueRange{allocated});
    rewriter.eraseOp(op);

    return success();
  }
};

struct SimSetInputOpLowering : public ModelAwarePattern<arc::SimSetInputOp> {
  using ModelAwarePattern::ModelAwarePattern;

  LogicalResult
  matchAndRewrite(arc::SimSetInputOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto modelIt =
        modelInfo.find(cast<SimModelInstanceType>(op.getInstance().getType())
                           .getModel()
                           .getValue());
    ModelInfoMap &model = modelIt->second;

    auto portIt = model.states.find(op.getInput());
    if (portIt == model.states.end()) {
      // If the port is not found in the state, it means the model does not
      // actually use it. Thus this operation is a no-op.
      rewriter.eraseOp(op);
      return success();
    }

    StateInfo &port = portIt->second;
    Value statePtr = createPtrToPortState(rewriter, op.getLoc(),
                                          adaptor.getInstance(), port);
    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, adaptor.getValue(),
                                               statePtr);

    return success();
  }
};

struct SimGetPortOpLowering : public ModelAwarePattern<arc::SimGetPortOp> {
  using ModelAwarePattern::ModelAwarePattern;

  LogicalResult
  matchAndRewrite(arc::SimGetPortOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto modelIt =
        modelInfo.find(cast<SimModelInstanceType>(op.getInstance().getType())
                           .getModel()
                           .getValue());
    ModelInfoMap &model = modelIt->second;

    auto type = typeConverter->convertType(op.getValue().getType());
    if (!type)
      return failure();
    auto portIt = model.states.find(op.getPort());
    if (portIt == model.states.end()) {
      // If the port is not found in the state, it means the model does not
      // actually set it. Thus this operation returns 0.
      rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(op, type, 0);
      return success();
    }

    StateInfo &port = portIt->second;
    Value statePtr = createPtrToPortState(rewriter, op.getLoc(),
                                          adaptor.getInstance(), port);
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, type, statePtr);

    return success();
  }
};

struct SimStepOpLowering : public ModelAwarePattern<arc::SimStepOp> {
  using ModelAwarePattern::ModelAwarePattern;

  LogicalResult
  matchAndRewrite(arc::SimStepOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto modelName = cast<SimModelInstanceType>(op.getInstance().getType())
                         .getModel()
                         .getValue();

    auto tracer = tracerMap.find(modelName);

    if (tracer != tracerMap.end())
      tracer->second.buildStepCall(adaptor.getInstance(), op.getLoc(),
                                   rewriter);

    StringAttr evalFunc =
        rewriter.getStringAttr(evalSymbolFromModelName(modelName));
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, mlir::TypeRange(), evalFunc,
                                              adaptor.getInstance());

    return success();
  }
};

/// Lowers SimEmitValueOp to a printf call. The integer will be printed in its
/// entirety if it is of size up to size_t, and explicitly truncated otherwise.
/// This pattern will mutate the global module.
struct SimEmitValueOpLowering
    : public OpConversionPattern<arc::SimEmitValueOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arc::SimEmitValueOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto valueType = dyn_cast<IntegerType>(adaptor.getValue().getType());
    if (!valueType)
      return failure();

    Location loc = op.getLoc();

    ModuleOp moduleOp = op->getParentOfType<ModuleOp>();
    if (!moduleOp)
      return failure();

    // Cast the value to a size_t.
    // FIXME: like the rest of MLIR, this assumes sizeof(intptr_t) ==
    // sizeof(size_t) on the target architecture.
    Value toPrint = adaptor.getValue();
    DataLayout layout = DataLayout::closest(op);
    llvm::TypeSize sizeOfSizeT =
        layout.getTypeSizeInBits(rewriter.getIndexType());
    assert(!sizeOfSizeT.isScalable() &&
           sizeOfSizeT.getFixedValue() <= std::numeric_limits<unsigned>::max());
    bool truncated = false;
    if (valueType.getWidth() > sizeOfSizeT) {
      toPrint = LLVM::TruncOp::create(
          rewriter, loc,
          IntegerType::get(getContext(), sizeOfSizeT.getFixedValue()), toPrint);
      truncated = true;
    } else if (valueType.getWidth() < sizeOfSizeT)
      toPrint = LLVM::ZExtOp::create(
          rewriter, loc,
          IntegerType::get(getContext(), sizeOfSizeT.getFixedValue()), toPrint);

    // Lookup of create printf function symbol.
    auto printfFunc = LLVM::lookupOrCreateFn(
        rewriter, moduleOp, "printf", LLVM::LLVMPointerType::get(getContext()),
        LLVM::LLVMVoidType::get(getContext()), true);
    if (failed(printfFunc))
      return printfFunc;

    // Insert the format string if not already available.
    SmallString<16> formatStrName{"_arc_sim_emit_"};
    formatStrName.append(truncated ? "trunc_" : "full_");
    formatStrName.append(adaptor.getValueName());
    LLVM::GlobalOp formatStrGlobal;
    if (!(formatStrGlobal =
              moduleOp.lookupSymbol<LLVM::GlobalOp>(formatStrName))) {
      ConversionPatternRewriter::InsertionGuard insertGuard(rewriter);

      SmallString<16> formatStr = adaptor.getValueName();
      formatStr.append(" = ");
      if (truncated)
        formatStr.append("(truncated) ");
      formatStr.append("%zx\n");
      SmallVector<char> formatStrVec{formatStr.begin(), formatStr.end()};
      formatStrVec.push_back(0);

      rewriter.setInsertionPointToStart(moduleOp.getBody());
      auto globalType =
          LLVM::LLVMArrayType::get(rewriter.getI8Type(), formatStrVec.size());
      formatStrGlobal = LLVM::GlobalOp::create(
          rewriter, loc, globalType, /*isConstant=*/true,
          LLVM::Linkage::Internal,
          /*name=*/formatStrName, rewriter.getStringAttr(formatStrVec),
          /*alignment=*/0);
    }

    Value formatStrGlobalPtr =
        LLVM::AddressOfOp::create(rewriter, loc, formatStrGlobal);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, printfFunc.value(), ValueRange{formatStrGlobalPtr, toPrint});

    return success();
  }
};

} // namespace

static LogicalResult convert(arc::ExecuteOp op, arc::ExecuteOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter,
                             const TypeConverter &converter) {
  // Convert the argument types in the body blocks.
  if (failed(rewriter.convertRegionTypes(&op.getBody(), converter)))
    return failure();

  // Split the block at the current insertion point such that we can branch into
  // the `arc.execute` body region, and have `arc.output` branch back to the
  // point after the `arc.execute`.
  auto *blockBefore = rewriter.getInsertionBlock();
  auto *blockAfter =
      rewriter.splitBlock(blockBefore, rewriter.getInsertionPoint());

  // Branch to the entry block.
  rewriter.setInsertionPointToEnd(blockBefore);
  mlir::cf::BranchOp::create(rewriter, op.getLoc(), &op.getBody().front(),
                             adaptor.getInputs());

  // Make all `arc.output` terminators branch to the block after the
  // `arc.execute` op.
  for (auto &block : op.getBody()) {
    auto outputOp = dyn_cast<arc::OutputOp>(block.getTerminator());
    if (!outputOp)
      continue;
    rewriter.setInsertionPointToEnd(&block);
    rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(outputOp, blockAfter,
                                                    outputOp.getOperands());
  }

  // Inline the body region between the before and after blocks.
  rewriter.inlineRegionBefore(op.getBody(), blockAfter);

  // Add arguments to the block after the `arc.execute`, replace the op's
  // results with the arguments, then perform block signature conversion.
  SmallVector<Value> args;
  args.reserve(op.getNumResults());
  for (auto result : op.getResults())
    args.push_back(blockAfter->addArgument(result.getType(), result.getLoc()));
  rewriter.replaceOp(op, args);
  auto conversion = converter.convertBlockSignature(blockAfter);
  if (!conversion)
    return failure();
  rewriter.applySignatureConversion(blockAfter, *conversion, &converter);
  return success();
}

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {

struct LowerArcToLLVMPass
    : public circt::impl::LowerArcToLLVMBase<LowerArcToLLVMPass> {
  void runOnOperation() override;
};
} // namespace

void LowerArcToLLVMPass::runOnOperation() {
  // Replace any `i0` values with an `hw.constant 0 : i0` to avoid later issues
  // in LLVM conversion.
  {
    DenseMap<Region *, hw::ConstantOp> zeros;
    getOperation().walk([&](Operation *op) {
      if (op->hasTrait<OpTrait::ConstantLike>())
        return;
      for (auto result : op->getResults()) {
        auto type = dyn_cast<IntegerType>(result.getType());
        if (!type || type.getWidth() != 0)
          continue;
        auto *region = op->getParentRegion();
        auto &zero = zeros[region];
        if (!zero) {
          auto builder = OpBuilder::atBlockBegin(&region->front());
          zero = hw::ConstantOp::create(builder, result.getLoc(),
                                        APInt::getZero(0));
        }
        result.replaceAllUsesWith(zero);
      }
    });
  }

  auto globalTraceHelpers = GlobalTraceHelpers(&getContext());
  llvm::SmallDenseMap<StringRef, ModelTraceHelpers> modelTraceHelpers;
  // Find models with trace taps
  SmallVector<ModelOp> traceableModels;
  for (auto modelOp : getOperation().getBody()->getOps<arc::ModelOp>()) {
    if (modelOp.getTraceTapsAttr() && !modelOp.getTraceTapsAttr().empty())
      traceableModels.push_back(modelOp);
  }

  {
    // Create the trace library registration, even if there are
    // no traceable models.
    ImplicitLocOpBuilder globalBuilder(getOperation().getLoc(), getOperation());
    globalBuilder.setInsertionPointToStart(getOperation().getBody());
    globalTraceHelpers.buildModuleGlobals(globalBuilder,
                                          traceableModels.size());
    globalBuilder.setInsertionPointToStart(getOperation().getBody());
    if (failed(globalTraceHelpers.buildAppendFunctions(globalBuilder,
                                                       traceableModels))) {
      signalPassFailure();
      return;
    }
  }

  // Build all the model specific tracing fluff.
  for (auto modelOp : traceableModels) {
    OpBuilder builder(getOperation());
    builder.setInsertionPointToStart(getOperation().getBody());
    builder.setInsertionPointToStart(getOperation().getBody());
    ModelTraceHelpers modelTrace(globalTraceHelpers, modelOp);
    modelTrace.buildStaticModuleStructs(builder);
    modelTraceHelpers.insert({modelOp.getNameAttr(), std::move(modelTrace)});
  }

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
  ConversionPatternSet patterns(&getContext(), converter);

  // MLIR patterns.
  populateSCFToControlFlowConversionPatterns(patterns);
  populateFuncToLLVMConversionPatterns(converter, patterns);
  cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
  arith::populateArithToLLVMConversionPatterns(converter, patterns);
  index::populateIndexToLLVMConversionPatterns(converter, patterns);
  populateAnyFunctionOpInterfaceTypeConversionPattern(patterns, converter);

  // CIRCT patterns.
  DenseMap<std::pair<Type, ArrayAttr>, LLVM::GlobalOp> constAggregateGlobalsMap;
  populateHWToLLVMConversionPatterns(converter, patterns, globals,
                                     constAggregateGlobalsMap);
  populateHWToLLVMTypeConversions(converter);
  populateCombToArithConversionPatterns(converter, patterns);
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
    ClockInvOpLowering,
    MemoryReadOpLowering,
    MemoryWriteOpLowering,
    ModelOpLowering,
    ReplaceOpWithInputPattern<seq::ToClockOp>,
    ReplaceOpWithInputPattern<seq::FromClockOp>,
    SeqConstClockLowering,
    SimEmitValueOpLowering,
    StateReadOpLowering,
    StorageGetOpLowering,
    ZeroCountOpLowering
  >(converter, &getContext());

  patterns.add<StateWriteOpLowering>(
      converter, &getContext(), modelTraceHelpers);
  // clang-format on
  patterns.add<ExecuteOp>(convert);

  SmallVector<ModelInfo> models;
  if (failed(collectModels(getOperation(), models))) {
    signalPassFailure();
    return;
  }

  llvm::DenseMap<StringRef, ModelInfoMap> modelMap(models.size());
  for (ModelInfo &modelInfo : models) {
    llvm::DenseMap<StringRef, StateInfo> states(modelInfo.states.size());
    for (StateInfo &stateInfo : modelInfo.states)
      states.insert({stateInfo.name, stateInfo});
    modelMap.insert(
        {modelInfo.name,
         ModelInfoMap{modelInfo.numStateBytes, std::move(states),
                      modelInfo.initialFnSym, modelInfo.finalFnSym}});
  }

  patterns.add<SimInstantiateOpLowering, SimSetInputOpLowering,
               SimGetPortOpLowering, SimStepOpLowering>(
      converter, &getContext(), modelMap, modelTraceHelpers);

  // Apply the conversion.
  ConversionConfig config;
  config.allowPatternRollback = false;
  if (failed(applyFullConversion(getOperation(), target, std::move(patterns),
                                 config)))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> circt::createLowerArcToLLVMPass() {
  return std::make_unique<LowerArcToLLVMPass>();
}
