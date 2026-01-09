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
#include "circt/Dialect/Arc/Runtime/Common.h"
#include "circt/Dialect/Arc/Runtime/JITBind.h"
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
#include "llvm/Support/FormatVariadic.h"

#include <cstddef>

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

struct StateWriteOpLowering : public OpConversionPattern<arc::StateWriteOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arc::StateWriteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (adaptor.getCondition()) {
      rewriter.replaceOpWithNewOp<scf::IfOp>(
          op, adaptor.getCondition(), [&](auto &builder, auto loc) {
            LLVM::StoreOp::create(builder, loc, adaptor.getValue(),
                                  adaptor.getState());
            scf::YieldOp::create(builder, loc);
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
                    llvm::DenseMap<StringRef, ModelInfoMap> &modelInfo)
      : OpConversionPattern<OpTy>(typeConverter, context),
        modelInfo(modelInfo) {}

protected:
  Value createPtrToPortState(ConversionPatternRewriter &rewriter, Location loc,
                             Value state, const StateInfo &port) const {
    MLIRContext *ctx = rewriter.getContext();
    return LLVM::GEPOp::create(rewriter, loc, LLVM::LLVMPointerType::get(ctx),
                               IntegerType::get(ctx, 8), state,
                               LLVM::GEPArg(port.offset));
  }

  llvm::DenseMap<StringRef, ModelInfoMap> &modelInfo;
};

/// Lowers SimInstantiateOp to a malloc and memset call. This pattern will
/// mutate the global module.
struct SimInstantiateOpLowering
    : public ModelAwarePattern<arc::SimInstantiateOp> {
  using ModelAwarePattern::ModelAwarePattern;

  LogicalResult
  matchAndRewrite(arc::SimInstantiateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto modelIt = modelInfo.find(
        cast<SimModelInstanceType>(op.getBody().getArgument(0).getType())
            .getModel()
            .getValue());
    ModelInfoMap &model = modelIt->second;

    bool useRuntime = op.getRuntimeModel().has_value();

    ModuleOp moduleOp = op->getParentOfType<ModuleOp>();
    if (!moduleOp)
      return failure();

    ConversionPatternRewriter::InsertionGuard guard(rewriter);

    // FIXME: like the rest of MLIR, this assumes sizeof(intptr_t) ==
    // sizeof(size_t) on the target architecture.
    Type convertedIndex = typeConverter->convertType(rewriter.getIndexType());
    Location loc = op.getLoc();
    Value allocated;

    if (useRuntime) {
      // The instance is using the runtime library
      auto ptrTy = LLVM::LLVMPointerType::get(getContext());

      Value runtimeArgs;
      // If present, materialize the runtime argument string on the stack
      if (op.getRuntimeArgs().has_value()) {
        SmallVector<int8_t> argStringVec(op.getRuntimeArgsAttr().begin(),
                                         op.getRuntimeArgsAttr().end());
        argStringVec.push_back('\0');
        auto strAttr = mlir::DenseElementsAttr::get(
            mlir::RankedTensorType::get({(int64_t)argStringVec.size()},
                                        rewriter.getI8Type()),
            llvm::ArrayRef(argStringVec));

        auto arrayCst = LLVM::ConstantOp::create(
            rewriter, loc,
            LLVM::LLVMArrayType::get(rewriter.getI8Type(), argStringVec.size()),
            strAttr);
        auto cst1 = LLVM::ConstantOp::create(rewriter, loc,
                                             rewriter.getI32IntegerAttr(1));
        runtimeArgs = LLVM::AllocaOp::create(rewriter, loc, ptrTy,
                                             arrayCst.getType(), cst1);
        LLVM::LifetimeStartOp::create(rewriter, loc, runtimeArgs);
        LLVM::StoreOp::create(rewriter, loc, arrayCst, runtimeArgs);
      } else {
        runtimeArgs = LLVM::ZeroOp::create(rewriter, loc, ptrTy).getResult();
      }
      // Call the state allocation function
      auto rtModelPtr = LLVM::AddressOfOp::create(rewriter, loc, ptrTy,
                                                  op.getRuntimeModelAttr())
                            .getResult();
      allocated =
          LLVM::CallOp::create(rewriter, loc, {ptrTy},
                               runtime::APICallbacks::symNameAllocInstance,
                               {rtModelPtr, runtimeArgs})
              .getResult();

      if (op.getRuntimeArgs().has_value())
        LLVM::LifetimeEndOp::create(rewriter, loc, runtimeArgs);

    } else {
      // The instance is not using the runtime library
      FailureOr<LLVM::LLVMFuncOp> mallocFunc =
          LLVM::lookupOrCreateMallocFn(rewriter, moduleOp, convertedIndex);
      if (failed(mallocFunc))
        return mallocFunc;

      Value numStateBytes = LLVM::ConstantOp::create(
          rewriter, loc, convertedIndex, model.numStateBytes);
      allocated = LLVM::CallOp::create(rewriter, loc, mallocFunc.value(),
                                       ValueRange{numStateBytes})
                      .getResult();
      Value zero =
          LLVM::ConstantOp::create(rewriter, loc, rewriter.getI8Type(), 0);
      LLVM::MemsetOp::create(rewriter, loc, allocated, zero, numStateBytes,
                             false);
    }

    // Call the model's 'initial' function if present.
    if (model.initialFnSymbol) {
      auto initialFnType = LLVM::LLVMFunctionType::get(
          LLVM::LLVMVoidType::get(op.getContext()),
          {LLVM::LLVMPointerType::get(op.getContext())});
      LLVM::CallOp::create(rewriter, loc, initialFnType, model.initialFnSymbol,
                           ValueRange{allocated});
    }

    // Execute the body.
    rewriter.inlineBlockBefore(&adaptor.getBody().getBlocks().front(), op,
                               {allocated});

    // Call the model's 'final' function if present.
    if (model.finalFnSymbol) {
      auto finalFnType = LLVM::LLVMFunctionType::get(
          LLVM::LLVMVoidType::get(op.getContext()),
          {LLVM::LLVMPointerType::get(op.getContext())});
      LLVM::CallOp::create(rewriter, loc, finalFnType, model.finalFnSymbol,
                           ValueRange{allocated});
    }

    if (useRuntime) {
      LLVM::CallOp::create(rewriter, loc, TypeRange{},
                           runtime::APICallbacks::symNameDeleteInstance,
                           {allocated});
    } else {
      FailureOr<LLVM::LLVMFuncOp> freeFunc =
          LLVM::lookupOrCreateFreeFn(rewriter, moduleOp);
      if (failed(freeFunc))
        return freeFunc;

      LLVM::CallOp::create(rewriter, loc, freeFunc.value(),
                           ValueRange{allocated});
    }

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
    StringRef modelName = cast<SimModelInstanceType>(op.getInstance().getType())
                              .getModel()
                              .getValue();

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

    SmallVector<Value> printfVariadicArgs;
    SmallString<16> printfFormatStr;
    int remainingBits = valueType.getWidth();
    Value value = adaptor.getValue();

    // Assumes the target platform uses 64bit for long long ints (%llx
    // formatter).
    constexpr llvm::StringRef intFormatter = "llx";
    auto intType = IntegerType::get(getContext(), 64);
    Value shiftValue = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getIntegerAttr(valueType, intType.getWidth()));

    if (valueType.getWidth() < intType.getWidth()) {
      int width = llvm::divideCeil(valueType.getWidth(), 4);
      printfFormatStr = llvm::formatv("%0{0}{1}", width, intFormatter);
      printfVariadicArgs.push_back(
          LLVM::ZExtOp::create(rewriter, loc, intType, value));
    } else {
      // Process the value in 64 bit chunks, starting from the least significant
      // bits. Since we append chunks in low-to-high order, we reverse the
      // vector to print them in the correct high-to-low order.
      int otherChunkWidth = intType.getWidth() / 4;
      int firstChunkWidth =
          llvm::divideCeil(valueType.getWidth() % intType.getWidth(), 4);
      if (firstChunkWidth == 0) { // print the full 64-bit hex or a subset.
        firstChunkWidth = otherChunkWidth;
      }

      std::string firstChunkFormat =
          llvm::formatv("%0{0}{1}", firstChunkWidth, intFormatter);
      std::string otherChunkFormat =
          llvm::formatv("%0{0}{1}", otherChunkWidth, intFormatter);

      for (int i = 0; remainingBits > 0; ++i) {
        // Append 64-bit chunks to the printf arguments, in low-to-high
        // order. The integer is printed in hex format with zero padding.
        printfVariadicArgs.push_back(
            LLVM::TruncOp::create(rewriter, loc, intType, value));

        // Zero-padded format specifier for fixed width, e.g. %01llx for 4 bits.
        printfFormatStr.append(i == 0 ? firstChunkFormat : otherChunkFormat);

        value =
            LLVM::LShrOp::create(rewriter, loc, value, shiftValue).getResult();
        remainingBits -= intType.getWidth();
      }
    }

    // Lookup of create printf function symbol.
    auto printfFunc = LLVM::lookupOrCreateFn(
        rewriter, moduleOp, "printf", LLVM::LLVMPointerType::get(getContext()),
        LLVM::LLVMVoidType::get(getContext()), true);
    if (failed(printfFunc))
      return printfFunc;

    // Insert the format string if not already available.
    SmallString<16> formatStrName{"_arc_sim_emit_"};
    formatStrName.append(adaptor.getValueName());
    LLVM::GlobalOp formatStrGlobal;
    if (!(formatStrGlobal =
              moduleOp.lookupSymbol<LLVM::GlobalOp>(formatStrName))) {
      ConversionPatternRewriter::InsertionGuard insertGuard(rewriter);

      SmallString<16> formatStr = adaptor.getValueName();
      formatStr.append(" = ");
      formatStr.append(printfFormatStr);
      formatStr.append("\n");
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

    // Add the format string to the end, and reverse the vector to print them in
    // the correct high-to-low order with the format string at the beginning.
    printfVariadicArgs.push_back(formatStrGlobalPtr);
    std::reverse(printfVariadicArgs.begin(), printfVariadicArgs.end());

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, printfFunc.value(),
                                              printfVariadicArgs);

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
// Runtime Implementation
//===----------------------------------------------------------------------===//

struct RuntimeModelOpLowering
    : public OpConversionPattern<arc::RuntimeModelOp> {
  using OpConversionPattern::OpConversionPattern;

  static constexpr uint64_t runtimeApiVersion = ARC_RUNTIME_API_VERSION;

  // Create a global LLVM struct containing the RuntimeModel metadata
  LogicalResult
  matchAndRewrite(arc::RuntimeModelOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    auto modelInfoStructType = LLVM::LLVMStructType::getLiteral(
        getContext(), {rewriter.getI64Type(), rewriter.getI64Type(),
                       LLVM::LLVMPointerType::get(getContext())});
    static_assert(sizeof(ArcRuntimeModelInfo) == 24 &&
                  "Unexpected size of ArcRuntimeModelInfo struct");

    // Construct the Model Name String GlobalOp
    rewriter.setInsertionPoint(op);
    SmallVector<char, 16> modNameArray(op.getName().begin(),
                                       op.getName().end());
    modNameArray.push_back('\0');
    auto nameGlobalType =
        LLVM::LLVMArrayType::get(rewriter.getI8Type(), modNameArray.size());
    SmallString<16> globalSymName{"_arc_mod_name_"};
    globalSymName.append(op.getName());
    auto nameGlobal = LLVM::GlobalOp::create(
        rewriter, op.getLoc(), nameGlobalType, /*isConstant=*/true,
        LLVM::Linkage::Internal,
        /*name=*/globalSymName, rewriter.getStringAttr(modNameArray),
        /*alignment=*/0);

    // Construct the Model Info Struct GlobalOp
    // Note: The struct is supposed to be constant at runtime, but contains the
    // relocatable address of another symbol, so it should not be placed in the
    // "rodata" section.
    auto modInfoGlobalOp =
        LLVM::GlobalOp::create(rewriter, op.getLoc(), modelInfoStructType,
                               /*isConstant=*/false, LLVM::Linkage::External,
                               op.getSymName(), Attribute{});

    // Struct Initializer
    Region &initRegion = modInfoGlobalOp.getInitializerRegion();
    Block *initBlock = rewriter.createBlock(&initRegion);
    rewriter.setInsertionPointToStart(initBlock);
    auto apiVersionCst = LLVM::ConstantOp::create(
        rewriter, op.getLoc(), rewriter.getI64IntegerAttr(runtimeApiVersion));
    auto numStateBytesCst = LLVM::ConstantOp::create(rewriter, op.getLoc(),
                                                     op.getNumStateBytesAttr());
    auto nameAddr =
        LLVM::AddressOfOp::create(rewriter, op.getLoc(), nameGlobal);
    Value initStruct =
        LLVM::PoisonOp::create(rewriter, op.getLoc(), modelInfoStructType);

    // Field: uint64_t apiVersion
    initStruct = LLVM::InsertValueOp::create(
        rewriter, op.getLoc(), initStruct, apiVersionCst, ArrayRef<int64_t>{0});
    static_assert(offsetof(ArcRuntimeModelInfo, apiVersion) == 0,
                  "Unexpected offset of field apiVersion");
    // Field: uint64_t numStateBytes
    initStruct =
        LLVM::InsertValueOp::create(rewriter, op.getLoc(), initStruct,
                                    numStateBytesCst, ArrayRef<int64_t>{1});
    static_assert(offsetof(ArcRuntimeModelInfo, numStateBytes) == 8,
                  "Unexpected offset of field numStateBytes");
    // Field: const char *modelName
    initStruct = LLVM::InsertValueOp::create(rewriter, op.getLoc(), initStruct,
                                             nameAddr, ArrayRef<int64_t>{2});
    static_assert(offsetof(ArcRuntimeModelInfo, modelName) == 16,
                  "Unexpected offset of field modelName");

    LLVM::ReturnOp::create(rewriter, op.getLoc(), initStruct);

    rewriter.replaceOp(op, modInfoGlobalOp);
    return success();
  }
};

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
  populateHWToLLVMTypeConversions(converter);
  std::optional<HWToLLVMArraySpillCache> spillCacheOpt =
      HWToLLVMArraySpillCache();
  {
    OpBuilder spillBuilder(getOperation());
    spillCacheOpt->spillNonHWOps(spillBuilder, converter, getOperation());
  }
  populateHWToLLVMConversionPatterns(converter, patterns, globals,
                                     constAggregateGlobalsMap, spillCacheOpt);

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
    RuntimeModelOpLowering,
    SeqConstClockLowering,
    SimEmitValueOpLowering,
    StateReadOpLowering,
    StateWriteOpLowering,
    StorageGetOpLowering,
    ZeroCountOpLowering
  >(converter, &getContext());
  // clang-format on
  patterns.add<ExecuteOp>(convert);

  auto &modelInfo = getAnalysis<ModelInfoAnalysis>();
  llvm::DenseMap<StringRef, ModelInfoMap> modelMap(modelInfo.infoMap.size());
  for (auto &[_, modelInfo] : modelInfo.infoMap) {
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
      converter, &getContext(), modelMap);

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
