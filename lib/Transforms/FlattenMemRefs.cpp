//===- FlattenMemRefs.cpp - MemRef flattening pass --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the MemRef flattening pass.
//
//===----------------------------------------------------------------------===//

#include "circt/Support/LLVM.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"

namespace circt {
#define GEN_PASS_DEF_FLATTENMEMREF
#define GEN_PASS_DEF_FLATTENMEMREFCALLS
#include "circt/Transforms/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;

bool circt::isUniDimensional(MemRefType memref) {
  return memref.getShape().size() == 1;
}

/// A struct for maintaining function declarations which needs to be rewritten,
/// if they contain memref arguments that was flattened.
struct FunctionRewrite {
  func::FuncOp op;
  FunctionType type;
};

static std::atomic<unsigned> globalCounter(0);
static DenseMap<StringAttr, StringAttr> globalNameMap;

static MemRefType getFlattenedMemRefType(MemRefType type) {
  return MemRefType::get(SmallVector<int64_t>{type.getNumElements()},
                         type.getElementType());
}

static std::string getFlattenedMemRefName(StringAttr baseName,
                                          MemRefType type) {
  unsigned uniqueID = globalCounter++;
  return llvm::formatv("{0}_{1}x{2}_{3}", baseName, type.getNumElements(),
                       type.getElementType(), uniqueID);
}

// Flatten indices by generating the product of the i'th index and the [0:i-1]
// shapes, for each index, and then summing these.
static Value flattenIndices(ConversionPatternRewriter &rewriter, Operation *op,
                            ValueRange indices, MemRefType memrefType) {
  assert(memrefType.hasStaticShape() && "expected statically shaped memref");
  Location loc = op->getLoc();

  if (indices.empty()) {
    // Singleton memref (e.g. memref<i32>) - return 0.
    return rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0))
        .getResult();
  }

  Value finalIdx = indices.front();
  for (auto memIdx : llvm::enumerate(indices.drop_front())) {
    Value partialIdx = memIdx.value();
    int64_t indexMulFactor = 1;

    // Calculate the product of the i'th index and the [0:i-1] shape dims.
    for (unsigned i = memIdx.index() + 1; i < memrefType.getShape().size();
         ++i) {
      int64_t dimSize = memrefType.getShape()[i];
      indexMulFactor *= dimSize;
    }

    // Multiply product by the current index operand.
    if (llvm::isPowerOf2_64(indexMulFactor)) {
      auto constant =
          rewriter
              .create<arith::ConstantOp>(
                  loc, rewriter.getIndexAttr(llvm::Log2_64(indexMulFactor)))
              .getResult();
      finalIdx =
          rewriter.create<arith::ShLIOp>(loc, finalIdx, constant).getResult();
    } else {
      auto constant = rewriter
                          .create<arith::ConstantOp>(
                              loc, rewriter.getIndexAttr(indexMulFactor))
                          .getResult();
      finalIdx =
          rewriter.create<arith::MulIOp>(loc, finalIdx, constant).getResult();
    }

    // Sum up with the prior lower dimension accessors.
    auto sumOp = rewriter.create<arith::AddIOp>(loc, finalIdx, partialIdx);
    finalIdx = sumOp.getResult();
  }
  return finalIdx;
}

static bool hasMultiDimMemRef(ValueRange values) {
  return llvm::any_of(values, [](Value v) {
    auto memref = dyn_cast<MemRefType>(v.getType());
    if (!memref)
      return false;
    return !isUniDimensional(memref);
  });
}

namespace {

struct LoadOpConversion : public OpConversionPattern<memref::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType type = op.getMemRefType();
    if (isUniDimensional(type) || !type.hasStaticShape() ||
        /*Already converted?*/ op.getIndices().size() == 1)
      return failure();
    Value finalIdx =
        flattenIndices(rewriter, op, adaptor.getIndices(), op.getMemRefType());
    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, adaptor.getMemref(),

                                                SmallVector<Value>{finalIdx});
    return success();
  }
};

struct StoreOpConversion : public OpConversionPattern<memref::StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType type = op.getMemRefType();
    if (isUniDimensional(type) || !type.hasStaticShape() ||
        /*Already converted?*/ op.getIndices().size() == 1)
      return failure();
    Value finalIdx =
        flattenIndices(rewriter, op, adaptor.getIndices(), op.getMemRefType());
    rewriter.replaceOpWithNewOp<memref::StoreOp>(op, adaptor.getValue(),
                                                 adaptor.getMemref(),
                                                 SmallVector<Value>{finalIdx});
    return success();
  }
};

struct AllocOpConversion : public OpConversionPattern<memref::AllocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType type = op.getType();
    if (isUniDimensional(type) || !type.hasStaticShape())
      return failure();
    MemRefType newType = getFlattenedMemRefType(type);
    rewriter.replaceOpWithNewOp<memref::AllocOp>(op, newType);
    return success();
  }
};

struct AllocaOpConversion : public OpConversionPattern<memref::AllocaOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocaOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType type = op.getType();
    if (isUniDimensional(type) || !type.hasStaticShape())
      return failure();
    MemRefType newType = getFlattenedMemRefType(type);
    rewriter.replaceOpWithNewOp<memref::AllocaOp>(op, newType);
    return success();
  }
};

struct GlobalOpConversion : public OpConversionPattern<memref::GlobalOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::GlobalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType type = op.getType();
    if (isUniDimensional(type) || !type.hasStaticShape())
      return failure();
    MemRefType newType = getFlattenedMemRefType(type);

    auto cstAttr =
        llvm::dyn_cast_or_null<DenseElementsAttr>(op.getConstantInitValue());

    SmallVector<Attribute> flattenedVals;
    for (auto attr : cstAttr.getValues<Attribute>())
      flattenedVals.push_back(attr);

    auto newTypeAttr = TypeAttr::get(newType);
    auto newNameStr = getFlattenedMemRefName(op.getConstantAttrName(), type);
    auto newName = rewriter.getStringAttr(newNameStr);
    globalNameMap[op.getSymNameAttr()] = newName;

    RankedTensorType tensorType = RankedTensorType::get(
        {static_cast<int64_t>(flattenedVals.size())}, type.getElementType());
    auto newInitValue = DenseElementsAttr::get(tensorType, flattenedVals);

    rewriter.replaceOpWithNewOp<memref::GlobalOp>(
        op, newName, op.getSymVisibilityAttr(), newTypeAttr, newInitValue,
        op.getConstantAttr(), op.getAlignmentAttr());

    return success();
  }
};

struct GetGlobalOpConversion : public OpConversionPattern<memref::GetGlobalOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::GetGlobalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *symbolTableOp = op->getParentWithTrait<mlir::OpTrait::SymbolTable>();
    auto globalOp = dyn_cast_or_null<memref::GlobalOp>(
        SymbolTable::lookupSymbolIn(symbolTableOp, op.getNameAttr()));

    MemRefType type = globalOp.getType();
    if (isUniDimensional(type) || !type.hasStaticShape())
      return failure();

    MemRefType newType = getFlattenedMemRefType(type);
    auto originalName = globalOp.getSymNameAttr();
    auto newNameIt = globalNameMap.find(originalName);
    if (newNameIt == globalNameMap.end())
      return failure();
    auto newName = newNameIt->second;

    rewriter.replaceOpWithNewOp<memref::GetGlobalOp>(op, newType, newName);

    return success();
  }
};

struct ReshapeOpConversion : public OpConversionPattern<memref::ReshapeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value flattenedSource = rewriter.getRemappedValue(op.getSource());
    if (!flattenedSource)
      return failure();

    auto flattenedSrcType = cast<MemRefType>(flattenedSource.getType());
    if (isUniDimensional(flattenedSrcType) ||
        !flattenedSrcType.hasStaticShape()) {
      rewriter.replaceOp(op, flattenedSource);
      return success();
    }

    return failure();
  }
};

// A generic pattern which will replace an op with a new op of the same type
// but using the adaptor (type converted) operands.
template <typename TOp>
struct OperandConversionPattern : public OpConversionPattern<TOp> {
  using OpConversionPattern<TOp>::OpConversionPattern;
  using OpAdaptor = typename TOp::Adaptor;
  LogicalResult
  matchAndRewrite(TOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TOp>(op, op->getResultTypes(),
                                     adaptor.getOperands(), op->getAttrs());
    return success();
  }
};

// Cannot use OperandConversionPattern for branch op since the default builder
// doesn't provide a method for communicating block successors.
struct CondBranchOpConversion
    : public OpConversionPattern<mlir::cf::CondBranchOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::cf::CondBranchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::cf::CondBranchOp>(
        op, adaptor.getCondition(), adaptor.getTrueDestOperands(),
        adaptor.getFalseDestOperands(), /*branch_weights=*/nullptr,
        op.getTrueDest(), op.getFalseDest());
    return success();
  }
};

// Rewrites a call op signature to flattened types. If rewriteFunctions is set,
// will also replace the callee with a private definition of the called
// function of the updated signature.
struct CallOpConversion : public OpConversionPattern<func::CallOp> {
  CallOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                   bool rewriteFunctions = false)
      : OpConversionPattern(typeConverter, context),
        rewriteFunctions(rewriteFunctions) {}

  LogicalResult
  matchAndRewrite(func::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<Type> convResTypes;
    if (typeConverter->convertTypes(op.getResultTypes(), convResTypes).failed())
      return failure();
    auto newCallOp = rewriter.create<func::CallOp>(
        op.getLoc(), adaptor.getCallee(), convResTypes, adaptor.getOperands());

    if (!rewriteFunctions) {
      rewriter.replaceOp(op, newCallOp);
      return success();
    }

    // Override any definition corresponding to the updated signature.
    // It is up to users of this pass to define how these rewritten functions
    // are to be implemented.
    rewriter.setInsertionPoint(op->getParentOfType<func::FuncOp>());
    auto *calledFunction = dyn_cast<CallOpInterface>(*op).resolveCallable();
    FunctionType funcType = FunctionType::get(
        op.getContext(), newCallOp.getOperandTypes(), convResTypes);
    func::FuncOp newFuncOp;
    if (calledFunction)
      newFuncOp = rewriter.replaceOpWithNewOp<func::FuncOp>(
          calledFunction, op.getCallee(), funcType);
    else
      newFuncOp =
          rewriter.create<func::FuncOp>(op.getLoc(), op.getCallee(), funcType);
    newFuncOp.setVisibility(SymbolTable::Visibility::Private);
    rewriter.replaceOp(op, newCallOp);

    return success();
  }

private:
  bool rewriteFunctions;
};

template <typename... TOp>
void addGenericLegalityConstraint(ConversionTarget &target) {
  (target.addDynamicallyLegalOp<TOp>([](TOp op) {
    return !hasMultiDimMemRef(op->getOperands()) &&
           !hasMultiDimMemRef(op->getResults());
  }),
   ...);
}

static void populateFlattenMemRefsLegality(ConversionTarget &target) {
  target.addLegalDialect<arith::ArithDialect>();
  target.addDynamicallyLegalOp<memref::AllocOp>(
      [](memref::AllocOp op) { return isUniDimensional(op.getType()); });
  target.addDynamicallyLegalOp<memref::AllocaOp>(
      [](memref::AllocaOp op) { return isUniDimensional(op.getType()); });
  target.addDynamicallyLegalOp<memref::StoreOp>(
      [](memref::StoreOp op) { return op.getIndices().size() == 1; });
  target.addDynamicallyLegalOp<memref::LoadOp>(
      [](memref::LoadOp op) { return op.getIndices().size() == 1; });
  target.addDynamicallyLegalOp<memref::GlobalOp>(
      [](memref::GlobalOp op) { return isUniDimensional(op.getType()); });
  target.addDynamicallyLegalOp<memref::GetGlobalOp>(
      [](memref::GetGlobalOp op) { return isUniDimensional(op.getType()); });
  addGenericLegalityConstraint<mlir::cf::CondBranchOp, mlir::cf::BranchOp,
                               func::CallOp, func::ReturnOp, memref::DeallocOp,
                               memref::CopyOp>(target);

  target.addDynamicallyLegalOp<func::FuncOp>([](func::FuncOp op) {
    auto argsConverted = llvm::none_of(op.getBlocks(), [](auto &block) {
      return hasMultiDimMemRef(block.getArguments());
    });

    auto resultsConverted = llvm::all_of(op.getResultTypes(), [](Type type) {
      if (auto memref = dyn_cast<MemRefType>(type))
        return isUniDimensional(memref);
      return true;
    });

    return argsConverted && resultsConverted;
  });
}

// Materializes a multidimensional memory to unidimensional memory by using a
// memref.collapse_shape operation.
// TODO: This is also possible for dynamically shaped memories.
static Value materializeCollapseShapeFlattening(OpBuilder &builder,
                                                MemRefType type,
                                                ValueRange inputs,
                                                Location loc) {
  assert(type.hasStaticShape() &&
         "Can only subview flatten memref's with static shape (for now...).");
  MemRefType sourceType = cast<MemRefType>(inputs[0].getType());
  int64_t memSize = sourceType.getNumElements();
  ArrayRef<int64_t> sourceShape = sourceType.getShape();
  ArrayRef<int64_t> targetShape = ArrayRef<int64_t>(memSize);

  // Build ReassociationIndices to collapse completely to 1D MemRef.
  auto indices = getReassociationIndicesForCollapse(sourceShape, targetShape);
  assert(indices.has_value() && "expected a valid collapse");

  // Generate the appropriate return type:
  return builder.create<memref::CollapseShapeOp>(loc, inputs[0],
                                                 indices.value());
}

static void populateTypeConversionPatterns(TypeConverter &typeConverter) {
  // Add default conversion for all types generically.
  typeConverter.addConversion([](Type type) { return type; });
  // Add specific conversion for memref types.
  typeConverter.addConversion([](MemRefType memref) {
    if (isUniDimensional(memref))
      return memref;
    return MemRefType::get(llvm::SmallVector<int64_t>{memref.getNumElements()},
                           memref.getElementType());
  });
}

struct FlattenMemRefPass
    : public circt::impl::FlattenMemRefBase<FlattenMemRefPass> {
public:
  void runOnOperation() override {

    auto *ctx = &getContext();
    TypeConverter typeConverter;
    populateTypeConversionPatterns(typeConverter);

    RewritePatternSet patterns(ctx);
    SetVector<StringRef> rewrittenCallees;
    patterns.add<LoadOpConversion, StoreOpConversion, AllocOpConversion,
                 AllocaOpConversion, GlobalOpConversion, GetGlobalOpConversion,
                 ReshapeOpConversion, OperandConversionPattern<func::ReturnOp>,
                 OperandConversionPattern<memref::DeallocOp>,
                 CondBranchOpConversion,
                 OperandConversionPattern<memref::DeallocOp>,
                 OperandConversionPattern<memref::CopyOp>, CallOpConversion>(
        typeConverter, ctx);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);

    ConversionTarget target(*ctx);
    populateFlattenMemRefsLegality(target);

    if (applyPartialConversion(getOperation(), target, std::move(patterns))
            .failed()) {
      signalPassFailure();
      return;
    }
  }
};

struct FlattenMemRefCallsPass
    : public circt::impl::FlattenMemRefCallsBase<FlattenMemRefCallsPass> {
public:
  void runOnOperation() override {
    auto *ctx = &getContext();
    TypeConverter typeConverter;
    populateTypeConversionPatterns(typeConverter);
    RewritePatternSet patterns(ctx);

    // Only run conversion on call ops within the body of the function. callee
    // functions are rewritten by rewriteFunctions=true. We do not use
    // populateFuncOpTypeConversionPattern to rewrite the function signatures,
    // since non-called functions should not have their types converted.
    // It is up to users of this pass to define how these rewritten functions
    // are to be implemented.
    patterns.add<CallOpConversion>(typeConverter, ctx,
                                   /*rewriteFunctions=*/true);

    ConversionTarget target(*ctx);
    target.addLegalDialect<memref::MemRefDialect, mlir::BuiltinDialect>();
    addGenericLegalityConstraint<func::CallOp>(target);
    addGenericLegalityConstraint<func::FuncOp>(target);

    // Add a target materializer to handle memory flattening through
    // memref.subview operations.
    typeConverter.addTargetMaterialization(materializeCollapseShapeFlattening);

    if (applyPartialConversion(getOperation(), target, std::move(patterns))
            .failed()) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

namespace circt {
std::unique_ptr<mlir::Pass> createFlattenMemRefPass() {
  return std::make_unique<FlattenMemRefPass>();
}

std::unique_ptr<mlir::Pass> createFlattenMemRefCallsPass() {
  return std::make_unique<FlattenMemRefCallsPass>();
}

} // namespace circt
