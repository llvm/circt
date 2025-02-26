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
#include "circt/Support/Namespace.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
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
      cast<IntegerType>(address.getType()).getWidth() + 1);
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
    auto memoryType = cast<MemoryType>(op.getMemory().getType());
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
        cast<MemoryType>(op.getMemory().getType()), rewriter);
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
    auto constTrue = rewriter.create<LLVM::ConstantOp>(op->getLoc(),
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
    return rewriter.create<LLVM::GEPOp>(loc, LLVM::LLVMPointerType::get(ctx),
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

    ModuleOp moduleOp = op->getParentOfType<ModuleOp>();
    if (!moduleOp)
      return failure();

    ConversionPatternRewriter::InsertionGuard guard(rewriter);

    // FIXME: like the rest of MLIR, this assumes sizeof(intptr_t) ==
    // sizeof(size_t) on the target architecture.
    Type convertedIndex = typeConverter->convertType(rewriter.getIndexType());

    FailureOr<LLVM::LLVMFuncOp> mallocFunc =
        LLVM::lookupOrCreateMallocFn(moduleOp, convertedIndex);
    if (failed(mallocFunc))
      return mallocFunc;

    FailureOr<LLVM::LLVMFuncOp> freeFunc = LLVM::lookupOrCreateFreeFn(moduleOp);
    if (failed(freeFunc))
      return freeFunc;

    Location loc = op.getLoc();
    Value numStateBytes = rewriter.create<LLVM::ConstantOp>(
        loc, convertedIndex, model.numStateBytes);
    Value allocated = rewriter
                          .create<LLVM::CallOp>(loc, mallocFunc.value(),
                                                ValueRange{numStateBytes})
                          .getResult();
    Value zero =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI8Type(), 0);
    rewriter.create<LLVM::MemsetOp>(loc, allocated, zero, numStateBytes, false);

    // Call the model's 'initial' function if present.
    if (model.initialFnSymbol) {
      auto initialFnType = LLVM::LLVMFunctionType::get(
          LLVM::LLVMVoidType::get(op.getContext()),
          {LLVM::LLVMPointerType::get(op.getContext())});
      rewriter.create<LLVM::CallOp>(loc, initialFnType, model.initialFnSymbol,
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
      rewriter.create<LLVM::CallOp>(loc, finalFnType, model.finalFnSymbol,
                                    ValueRange{allocated});
    }

    rewriter.create<LLVM::CallOp>(loc, freeFunc.value(), ValueRange{allocated});
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
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, std::nullopt, evalFunc,
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
      toPrint = rewriter.create<LLVM::TruncOp>(
          loc, IntegerType::get(getContext(), sizeOfSizeT.getFixedValue()),
          toPrint);
      truncated = true;
    } else if (valueType.getWidth() < sizeOfSizeT)
      toPrint = rewriter.create<LLVM::ZExtOp>(
          loc, IntegerType::get(getContext(), sizeOfSizeT.getFixedValue()),
          toPrint);

    // Lookup of create printf function symbol.
    auto printfFunc = LLVM::lookupOrCreateFn(
        moduleOp, "printf", LLVM::LLVMPointerType::get(getContext()),
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
      formatStrGlobal = rewriter.create<LLVM::GlobalOp>(
          loc, globalType, /*isConstant=*/true, LLVM::Linkage::Internal,
          /*name=*/formatStrName, rewriter.getStringAttr(formatStrVec),
          /*alignment=*/0);
    }

    Value formatStrGlobalPtr =
        rewriter.create<LLVM::AddressOfOp>(loc, formatStrGlobal);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, printfFunc.value(), ValueRange{formatStrGlobalPtr, toPrint});

    return success();
  }
};

} // namespace

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
    StateWriteOpLowering,
    StorageGetOpLowering,
    ZeroCountOpLowering
  >(converter, &getContext());
  // clang-format on

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
      converter, &getContext(), modelMap);

  // Apply the conversion.
  if (failed(applyFullConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> circt::createLowerArcToLLVMPass() {
  return std::make_unique<LowerArcToLLVMPass>();
}
