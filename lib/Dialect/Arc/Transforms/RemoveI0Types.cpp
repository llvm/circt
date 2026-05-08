//===- RemoveI0Types.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_REMOVEI0TYPES
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace arc;

namespace {
struct RemoveI0TypesPass
    : public arc::impl::RemoveI0TypesBase<RemoveI0TypesPass> {
  using RemoveI0TypesBase::RemoveI0TypesBase;
  void runOnOperation() override;
};

bool isI0(Type type) {
  auto intType = dyn_cast<IntegerType>(type);
  return intType && intType.getWidth() == 0;
}

bool isPoison(Type type) { return isa<OpaqueType>(type); }

SmallVector<Type> filterOutPoison(TypeRange range) {
  return llvm::filter_to_vector(range, [](Type t) { return !isPoison(t); });
}

SmallVector<Value> filterOutPoison(ValueRange range) {
  return llvm::filter_to_vector(range,
                                [](Value v) { return !isPoison(v.getType()); });
}

struct ConvertFunc : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const TypeConverter &converter = *getTypeConverter();
    FunctionType fty = op.getFunctionType();

    rewriter.setInsertionPointToStart(&op.getBody().front());
    TypeConverter::SignatureConversion sig(op.getNumArguments());
    for (auto [origIdx, type] : enumerate(fty.getInputs())) {
      Type newType = converter.convertType(type);
      if (!newType)
        return failure();
      if (isPoison(newType)) {
        Value cast = UnrealizedConversionCastOp::create(rewriter, op.getLoc(),
                                                        newType, ValueRange{})
                         .getResult(0);
        sig.remapInput(origIdx, cast);
        continue;
      }
      sig.addInputs(origIdx, newType);
    }

    if (failed(rewriter.convertRegionTypes(&op.getBody(), converter, &sig))) {
      return failure();
    }

    SmallVector<Type> filteredResultTypes;
    if (failed(converter.convertTypes(fty.getResults(), filteredResultTypes)))
      return failure();
    filteredResultTypes = filterOutPoison(filteredResultTypes);

    rewriter.modifyOpInPlace(op, [&]() {
      op.setFunctionType(FunctionType::get(
          getContext(), sig.getConvertedTypes(), filteredResultTypes));
    });

    return success();
  }
};

struct ConvertReturn : public OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> filteredOperands =
        filterOutPoison(adaptor.getOperands());
    rewriter.modifyOpInPlace(op, [&]() { op->setOperands(filteredOperands); });

    return success();
  }
};

struct ConvertCall : public OpConversionPattern<func::CallOp> {
  using OpConversionPattern<func::CallOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const TypeConverter &converter = *getTypeConverter();
    SmallVector<Value> filteredOperands =
        filterOutPoison(adaptor.getOperands());
    SmallVector<Type> resultTypes;
    if (failed(converter.convertTypes(op.getResultTypes(), resultTypes)))
      return failure();
    SmallVector<Type> filteredResultTypes = filterOutPoison(resultTypes);

    auto newOp =
        func::CallOp::create(rewriter, op.getLoc(), filteredResultTypes,
                             filteredOperands, op->getAttrs());

    auto newOpResultIt = newOp.result_begin();
    SmallVector<Value> results;
    for (auto type : resultTypes) {
      if (isPoison(type)) {
        results.push_back(UnrealizedConversionCastOp::create(
                              rewriter, op.getLoc(), type, ValueRange{})
                              .getResult(0));
      } else {
        results.push_back(*newOpResultIt++);
      }
    }
    rewriter.replaceOp(op, results);
    return success();
  }
};

struct ConvertArrayGet : public OpConversionPattern<hw::ArrayGetOp> {
  using OpConversionPattern<hw::ArrayGetOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(hw::ArrayGetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

struct ConvertArrayCreate : public OpConversionPattern<hw::ArrayCreateOp> {
  using OpConversionPattern<hw::ArrayCreateOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(hw::ArrayCreateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getInputs().front());
    return success();
  }
};

struct ConvertArrayInject : public OpConversionPattern<hw::ArrayInjectOp> {
  using OpConversionPattern<hw::ArrayInjectOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(hw::ArrayInjectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getElement());
    return success();
  }
};

struct HandleGenericOp : public ConversionPattern {
  HandleGenericOp(TypeConverter &converter, MLIRContext *context)
      : ConversionPattern(converter, MatchAnyOpTypeTag{},
                          /*benefit=*/0, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Any op returning i0 must be dead.
    if (llvm::any_of(op->getResults(),
                     [](Value v) { return isI0(v.getType()); })) {
      rewriter.eraseOp(op);
      return success();
    }

    // Otherwise just perform type replacement.
    auto result =
        convertOpResultTypes(op, operands, *getTypeConverter(), rewriter);
    if (failed(result))
      return failure();

    rewriter.replaceOp(op, *result);
    return success();
  }
};

struct ConvertAggregateConstant : public OpConversionPattern<hw::AggregateConstantOp> {
  using OpConversionPattern<hw::AggregateConstantOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(hw::AggregateConstantOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());

    // Recursively rewrite the attribute.
    Attribute newFields = rewriteArrayAttr(op.getFields(), op.getResult().getType());

    if (!isa<ArrayAttr>(newFields)) {
      // Scalar result -> becomes hw.constant.
      IntegerAttr attr = cast<IntegerAttr>(newFields);
      auto result = hw::ConstantOp::create(rewriter, op.getLoc(), resultType,
                                           attr);
      rewriter.replaceOp(op, result);
      return success();
    }

    // Composite result -> new aggregate_constant op.
    auto newOp = hw::AggregateConstantOp::create(rewriter, op.getLoc(), resultType, cast<ArrayAttr>(newFields));
    rewriter.replaceOp(op, newOp);
    return success();
  }

  Attribute rewriteArrayAttr(ArrayAttr array, Type type) const {
    if (getTypeConverter()->convertType(type) == type)
      return array;
    if (auto arrayType = dyn_cast<hw::ArrayType>(type); arrayType && arrayType.getNumElements() == 1) {
      return *array.begin();
    }

    // Collect the immediate subtypes. FieldIDTypeInterface is supported by
    // ArrayType, UnpackedArrayType, StructType, UnionType.
    auto fieldIdInterface = cast<hw::FieldIDTypeInterface>(type);
    SmallVector<Attribute> attrs;
    for (auto [index, attr] : llvm::enumerate(array)) {
      uint64_t fieldId = fieldIdInterface.getFieldID(index);
      Type subType = fieldIdInterface.getSubTypeByFieldID(fieldId).first;
      if (auto subArrayAttr = dyn_cast<ArrayAttr>(attr)) {
        attrs.push_back(rewriteArrayAttr(subArrayAttr, subType));
      } else {
        attrs.push_back(attr);
      }
    }
    return ArrayAttr::get(array.getContext(), attrs);
  }
};
} // namespace

void RemoveI0TypesPass::runOnOperation() {
  TypeConverter converter;
  ConversionTarget target(getContext());
  RewritePatternSet patterns(&getContext());

  converter.addConversion([](Type type) -> Type {
    if (isI0(type))
      return OpaqueType::get(StringAttr::get(type.getContext(), "arc"),
                             StringAttr::get(type.getContext(), "poison"));
    return type;
  });
  converter.addConversion([&converter](hw::ArrayType type) -> Type {
    if (type.getNumElements() == 1)
      return converter.convertType(type.getElementType());
    // Recursively apply type conversion to inner types.
    return hw::ArrayType::get(converter.convertType(type.getElementType()),
                              type.getNumElements());
  });

  // Composite types - recursively apply type conversion to inner types.
  converter.addConversion([&converter](hw::StructType type) {
    auto newMembers =
        map_to_vector(type.getElements(), [&](hw::StructType::FieldInfo field) {
          field.type = converter.convertType(field.type);
          return field;
        });
    return hw::StructType::get(type.getContext(), newMembers);
  });
  converter.addConversion([&converter](hw::UnionType type) {
    auto newMembers =
        map_to_vector(type.getElements(), [&](hw::UnionType::FieldInfo field) {
          field.type = converter.convertType(field.type);
          return field;
        });
    return hw::UnionType::get(type.getContext(), newMembers);
  });
  converter.addConversion([&converter](hw::TypeAliasType type) {
    return converter.convertType(type.getCanonicalType());
  });
  converter.addConversion([&converter](arc::StateType type) {
    return arc::StateType::get(converter.convertType(type.getType()));
  });

  target.markUnknownOpDynamicallyLegal(
      [&](Operation *op) { return converter.isLegal(op); });
  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp func) {
    FunctionType fty = func.getFunctionType();
    return converter.isLegal(fty.getInputs()) &&
           converter.isLegal(fty.getResults());
  });

  patterns.add<ConvertFunc, ConvertReturn, ConvertCall, ConvertArrayGet,
               ConvertArrayCreate, ConvertArrayInject, HandleGenericOp,
               ConvertAggregateConstant>(
      converter, &getContext());
  if (failed(
          applyFullConversion(getOperation(), target, std::move(patterns)))) {
    return signalPassFailure();
  }

  // Run an empty set of patterns through applyPatternsGreedily to perform
  // a poor-man's DCE.
  if (failed(applyPatternsGreedily(getOperation(),
                                   RewritePatternSet(&getContext())))) {
    return signalPassFailure();
  }
}
