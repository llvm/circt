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
#include "mlir/Dialect/ControlFlow/Transforms/StructuralTypeConversions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
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
#include "mlir/IR/DialectResourceBlobManager.h"

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

static Value getValueFromOpFoldResult(OpBuilder &builder, Location loc,
                                      OpFoldResult in) {
  if (auto attr = dyn_cast<Attribute>(in))
    return arith::ConstantIndexOp::create(builder, loc,
                                          cast<IntegerAttr>(attr).getInt());
  return cast<Value>(in);
}

static FailureOr<std::pair<Value, Value>>
getFlattenMemrefAndOffset(OpBuilder &builder, Location loc, Value source,
                          ValueRange indices) {
  auto sourceType = dyn_cast<MemRefType>(source.getType());
  if (!sourceType)
    return failure();

  int64_t sourceOffset;
  SmallVector<int64_t, 4> sourceStrides;
  if (failed(sourceType.getStridesAndOffset(sourceStrides, sourceOffset)))
    return failure();

  auto stridedMetadata =
      memref::ExtractStridedMetadataOp::create(builder, loc, source);

  auto typeBit = sourceType.getElementType().getIntOrFloatBitWidth();
  if (typeBit == 0)
    typeBit = 8;

  OpFoldResult linearizedIndices;
  memref::LinearizedMemRefInfo linearizedInfo;
  std::tie(linearizedInfo, linearizedIndices) =
      memref::getLinearizedMemRefOffsetAndSize(
          builder, loc, typeBit, typeBit,
          stridedMetadata.getConstifiedMixedOffset(),
          stridedMetadata.getConstifiedMixedSizes(),
          stridedMetadata.getConstifiedMixedStrides(),
          getAsOpFoldResult(indices));

  Value flatMemref = memref::ReinterpretCastOp::create(
      builder, loc, source, linearizedInfo.linearizedOffset,
      ArrayRef<OpFoldResult>{linearizedInfo.linearizedSize},
      ArrayRef<OpFoldResult>{builder.getIndexAttr(1)});

  return std::make_pair(flatMemref,
                        getValueFromOpFoldResult(builder, loc,
                                                 linearizedIndices));
}

struct ExpandShapeOpConversion
    : public OpConversionPattern<memref::ExpandShapeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::ExpandShapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Once memref types are flattened, expand_shape is a no-op.
    rewriter.replaceOp(op, adaptor.getSrc());
    return success();
  }
};

struct CollapseShapeOpConversion
    : public OpConversionPattern<memref::CollapseShapeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::CollapseShapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // After flattening, collapse is also a no-op.
    rewriter.replaceOp(op, adaptor.getSrc());
    return success();
  }
};

// Flatten indices by generating the product of the i'th index and the [0:i-1]
// shapes, for each index, and then summing these.
static Value flattenIndices(ConversionPatternRewriter &rewriter, Operation *op,
                            ValueRange indices, MemRefType memrefType) {
  assert(memrefType.hasStaticShape() && "expected statically shaped memref");
  Location loc = op->getLoc();

  if (indices.empty()) {
    // Singleton memref (e.g. memref<i32>) - return 0.
    return arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(0))
        .getResult();
  }

  Value finalIdx = arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(0))
                       .getResult();
  auto shape = memrefType.getShape();
  for (auto [dim, idx] : llvm::zip(shape, indices)) {
    auto dimVal =
        arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(dim))
            .getResult();
    finalIdx = arith::MulIOp::create(rewriter, loc, finalIdx, dimVal).getResult();
    finalIdx = arith::AddIOp::create(rewriter, loc, finalIdx, idx).getResult();
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
    auto flatMemrefTy = dyn_cast<MemRefType>(adaptor.getMemref().getType());
    if (!flatMemrefTy || !isUniDimensional(flatMemrefTy))
      return failure();
    Value finalIdx = flattenIndices(rewriter, op, adaptor.getIndices(), type);

    rewriter.replaceOpWithNewOp<memref::LoadOp>(
        op, adaptor.getMemref(), SmallVector<Value>{finalIdx});
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
    auto flatMemrefTy = dyn_cast<MemRefType>(adaptor.getMemref().getType());
    if (!flatMemrefTy || !isUniDimensional(flatMemrefTy))
      return failure();
    Value finalIdx = flattenIndices(rewriter, op, adaptor.getIndices(), type);

    rewriter.replaceOpWithNewOp<memref::StoreOp>(
        op, adaptor.getValue(), adaptor.getMemref(), SmallVector<Value>{finalIdx});
    return success();
  }
};

struct AllocOpConversion : public OpConversionPattern<memref::AllocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
matchAndRewrite(memref::AllocOp op, OpAdaptor adaptor,
                ConversionPatternRewriter &rewriter) const override {
  MemRefType type = op.getType();
  if (isUniDimensional(type) || !type.hasStaticShape())
    return failure();

  MemRefType newType = getFlattenedMemRefType(type);

  rewriter.replaceOpWithNewOp<memref::AllocOp>(
      op, newType,
      adaptor.getDynamicSizes(),
      adaptor.getSymbolOperands(),
      op.getAlignmentAttr());
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

    // Accept any ElementsAttr (DenseElementsAttr or DenseResourceElementsAttr, etc.)
    auto elements =
        llvm::dyn_cast_or_null<ElementsAttr>(op.getConstantInitValue());
    if (!elements)
      return rewriter.notifyMatchFailure(
          op, "memref.global has non-elements or missing constant initializer");

    // Build the flattened initializer type: 1-D tensor with same element type.
    auto oldShapedTy = llvm::dyn_cast<ShapedType>(elements.getType());
    if (!oldShapedTy || !oldShapedTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "initializer is not a static shaped type");

    int64_t nElts = oldShapedTy.getNumElements();
    RankedTensorType tensorType =
        RankedTensorType::get({nElts}, type.getElementType());

    // Produce a new initializer without iterating element-by-element.
    ElementsAttr newInitValue;
    if (auto dense = llvm::dyn_cast<DenseElementsAttr>(elements)) {
      // DenseElementsAttr supports reshape in your revision.
      newInitValue = dense.reshape(tensorType);
    } else if (auto res = llvm::dyn_cast<DenseResourceElementsAttr>(elements)) {
      // Resource-backed: reuse the same handle, just change the shaped type.
      newInitValue = DenseResourceElementsAttr::get(tensorType, res.getRawHandle());
    } else if (auto splat = llvm::dyn_cast<SplatElementsAttr>(elements)) {
      // Optional: preserve splat.
      newInitValue = SplatElementsAttr::get(tensorType, splat.getSplatValue<Attribute>());
    } else {
      return rewriter.notifyMatchFailure(op, "unsupported elements initializer kind");
    }

    auto newTypeAttr = TypeAttr::get(newType);
    auto newNameStr = getFlattenedMemRefName(op.getConstantAttrName(), type);
    auto newName = rewriter.getStringAttr(newNameStr);
    globalNameMap[op.getSymNameAttr()] = newName;

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

static Value materializeFlattenTo1D(OpBuilder &builder, MemRefType targetType,
                                    ValueRange inputs, Location loc);

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

struct SubViewOpConversion : public OpConversionPattern<memref::SubViewOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::SubViewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto sourceType = op.getSourceType();
    auto resultType = op.getType();
    if (!sourceType.hasStaticShape() || !resultType.hasStaticShape())
      return failure();
    if (isUniDimensional(sourceType))
      return failure();
    if (isUniDimensional(resultType))
      return failure();

    Value source = op.getSource();
    if (isa<BlockArgument>(source))
      source = memref::CastOp::create(rewriter, op.getLoc(), source.getType(),
                                      source);
    Value flatSource = materializeFlattenTo1D(
        rewriter, getFlattenedMemRefType(sourceType), ValueRange{source},
        op.getLoc());
    if (!flatSource)
      return failure();

    SmallVector<Value> offsets;
    offsets.reserve(op.getMixedOffsets().size());
    for (OpFoldResult o : op.getMixedOffsets())
      offsets.push_back(getValueFromOpFoldResult(rewriter, op.getLoc(), o));

    Value linearOffset = flattenIndices(rewriter, op, offsets, sourceType);
    MemRefType flatResultType = MemRefType::get(
        {resultType.getNumElements()}, resultType.getElementType(),
        StridedLayoutAttr::get(rewriter.getContext(), ShapedType::kDynamic,
                               {1}),
        resultType.getMemorySpace());

    Value flatSubview = memref::ReinterpretCastOp::create(
        rewriter, op.getLoc(), flatResultType, flatSource,
        linearOffset,
        ArrayRef<OpFoldResult>{
            rewriter.getIndexAttr(flatResultType.getNumElements())},
        ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)});
    rewriter.replaceOp(op, flatSubview);
    return success();
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
    auto newCallOp =
        func::CallOp::create(rewriter, op.getLoc(), adaptor.getCallee(),
                             convResTypes, adaptor.getOperands());

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
          func::FuncOp::create(rewriter, op.getLoc(), op.getCallee(), funcType);
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
  target.addDynamicallyLegalOp<memref::SubViewOp>([](memref::SubViewOp op) {
    auto srcType = dyn_cast<MemRefType>(op.getSource().getType());
    auto dstType = dyn_cast<MemRefType>(op.getType());
    return srcType && dstType && isUniDimensional(srcType) &&
           isUniDimensional(dstType);
  });
  addGenericLegalityConstraint<func::CallOp, func::ReturnOp, memref::DeallocOp,
                               memref::CopyOp, memref::ReinterpretCastOp>(target);

  target.addDynamicallyLegalOp<func::FuncOp>([](func::FuncOp op) {
    auto argsConverted = llvm::all_of(op.getArgumentTypes(), [](Type type) {
      if (auto memref = dyn_cast<MemRefType>(type))
        return isUniDimensional(memref);
      return true;
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

  Value collapsed =
      memref::CollapseShapeOp::create(builder, loc, inputs[0], indices.value());

  if (collapsed.getType() == type)
    return collapsed;

  return memref::CastOp::create(builder, loc, type, collapsed);
}

// Materializes a 1D flattened memref back into an identity-layout ranked memref
// using reinterpret_cast so view-like users (e.g. subview) can remain legal.
static Value materializeExpandShapeFromFlattened(OpBuilder &builder,
                                                 MemRefType targetType,
                                                 ValueRange inputs,
                                                 Location loc) {
  if (inputs.size() != 1)
    return {};

  auto sourceType = dyn_cast<MemRefType>(inputs.front().getType());
  if (!sourceType || !sourceType.hasStaticShape() || !targetType.hasStaticShape())
    return {};
  if (!isUniDimensional(sourceType))
    return {};
  if (sourceType.getNumElements() != targetType.getNumElements())
    return {};

  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides(targetType.getRank());
  auto shape = targetType.getShape();
  int64_t stride = 1;
  for (int64_t i = targetType.getRank() - 1; i >= 0; --i) {
    sizes.push_back(builder.getIndexAttr(shape[i]));
    strides[i] = builder.getIndexAttr(stride);
    stride *= shape[i];
  }
  std::reverse(sizes.begin(), sizes.end());

  return memref::ReinterpretCastOp::create(builder, loc, targetType,
                                           inputs.front(),
                                           builder.getIndexAttr(0), sizes,
                                           strides);
}

// Materialize a ranked memref to a 1D flattened memref type.
static Value materializeFlattenTo1D(OpBuilder &builder, MemRefType targetType,
                                    ValueRange inputs, Location loc) {
  if (inputs.size() != 1)
    return {};
  auto sourceType = dyn_cast<MemRefType>(inputs.front().getType());
  if (!sourceType || !sourceType.hasStaticShape())
    return {};
  if (!isUniDimensional(targetType))
    return {};
  if (sourceType.getNumElements() != targetType.getNumElements())
    return {};

  return memref::ReinterpretCastOp::create(
      builder, loc, targetType, inputs.front(), builder.getIndexAttr(0),
      ArrayRef<OpFoldResult>{builder.getIndexAttr(targetType.getNumElements())},
      ArrayRef<OpFoldResult>{builder.getIndexAttr(1)});
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
    typeConverter.addTargetMaterialization(materializeExpandShapeFromFlattened);
    typeConverter.addSourceMaterialization(materializeFlattenTo1D);

    RewritePatternSet patterns(ctx);
    SetVector<StringRef> rewrittenCallees;
    patterns.add<LoadOpConversion, StoreOpConversion, AllocOpConversion,
                 AllocaOpConversion, GlobalOpConversion, GetGlobalOpConversion,
                 ReshapeOpConversion, SubViewOpConversion,
                 ExpandShapeOpConversion,
		 CollapseShapeOpConversion,
		 OperandConversionPattern<func::ReturnOp>,
                 OperandConversionPattern<memref::DeallocOp>,
                 OperandConversionPattern<memref::DeallocOp>,
                 OperandConversionPattern<memref::CopyOp>,
                 OperandConversionPattern<memref::ReinterpretCastOp>,
                 CallOpConversion>(
        typeConverter, ctx);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);

    ConversionTarget target(*ctx);
    populateFlattenMemRefsLegality(target);
    mlir::cf::populateCFStructuralTypeConversionsAndLegality(typeConverter,
                                                             patterns, target);

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
