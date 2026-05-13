//===- RemoveI0Types.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcDialect.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/Arc/ArcTypes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/LLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

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

// Flattens a list of list of values into a list of values.
SmallVector<Value> flatten(ArrayRef<ValueRange> ranges) {
  SmallVector<Value> flat;
  for (auto range : ranges) {
    flat.insert(flat.end(), range.begin(), range.end());
  }
  return flat;
}

// Generic pattern for ops that are legalizable by flattening their remaining
// operands after type conversion.
template <typename T>
struct LegalizeGeneric : public OpConversionPattern<T> {
  using OpConversionPattern<T>::OpConversionPattern;
  using OneToNOpAdaptor = typename OpConversionPattern<T>::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(T op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const TypeConverter &converter = *this->getTypeConverter();
    auto result = convertOpResultTypes(op, flatten(adaptor.getOperands()),
                                       converter, rewriter);
    if (failed(result))
      return failure();

    // Map from old results to new results, assuming the results size may have
    // changed.
    Operation *newOp = *result;
    auto newOpResultIt = newOp->result_begin();
    SmallVector<Value> results;
    for (auto oldType : op->getResultTypes()) {
      if (!converter.convertType(oldType)) {
        results.push_back(nullptr);
      } else {
        results.push_back(*newOpResultIt++);
      }
    }
    assert(newOpResultIt == newOp->result_end() && "Didn't map all results!");
    rewriter.replaceOp(op, results);
    return success();
  }
};

// As above, but if any converted operand is empty (i.e. the operand was i0
// initially) or the number of results after conversion changes, then the op is
// erased. This is used for all ops where we don't explicitly know the legality
// of removing operands from its operand list.
struct ConvertGeneric : public ConversionPattern {
  ConvertGeneric(TypeConverter &converter, MLIRContext *context)
      : ConversionPattern(converter, MatchAnyOpTypeTag{},
                          /*benefit=*/0, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<ValueRange> operands,
                  ConversionPatternRewriter &rewriter) const override {
    const TypeConverter &converter = *getTypeConverter();

    // If any operand wasn't converted (empty range), the op must be dead.
    for (ValueRange range : operands) {
      if (range.empty()) {
        rewriter.eraseOp(op);
        return success();
      }
    }

    // If any result wasn't converted, the op must be dead.
    SmallVector<Type> resultTypes;
    if (failed(converter.convertTypes(op->getResultTypes(), resultTypes)))
      return failure();
    if (resultTypes.size() != op->getNumResults()) {
      rewriter.eraseOp(op);
      return success();
    }

    auto result =
        convertOpResultTypes(op, flatten(operands), converter, rewriter);
    if (failed(result))
      return failure();
    rewriter.replaceOp(op, *result);
    return success();
  }
};

// An array_get with i0 index just returns the array input, which will have been
// scalarized.
struct ConvertArrayGet : public OpConversionPattern<hw::ArrayGetOp> {
  using OpConversionPattern<hw::ArrayGetOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(hw::ArrayGetOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getIndex().empty()) {
      assert(adaptor.getInput().size() == 1);
      rewriter.replaceOp(op, adaptor.getInput().front());
      return success();
    }
    // Handled by ConvertGeneric.
    return failure();
  }
};

// Replaces array_create of a single element with the element.
struct ConvertArrayCreate : public OpConversionPattern<hw::ArrayCreateOp> {
  using OpConversionPattern<hw::ArrayCreateOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(hw::ArrayCreateOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getInputs().size() == 1) {
      rewriter.replaceOp(op, adaptor.getInputs().front());
      return success();
    }
    // Handled by ConvertGeneric.
    return failure();
  }
};

// Converts array_inject with i0 index to just return the element.
struct ConvertArrayInject : public OpConversionPattern<hw::ArrayInjectOp> {
  using OpConversionPattern<hw::ArrayInjectOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(hw::ArrayInjectOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getIndex().empty()) {
      rewriter.replaceOp(op, adaptor.getElement());
      return success();
    }
    // Handled by ConvertGeneric.
    return failure();
  }
};

// Converts an aggregate_constant by recursively rewriting its attribute.
struct ConvertAggregateConstant
    : public OpConversionPattern<hw::AggregateConstantOp> {
  using OpConversionPattern<hw::AggregateConstantOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(hw::AggregateConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());

    // Recursively rewrite the attribute.
    Attribute newFields =
        rewriteArrayAttr(op.getFields(), op.getResult().getType());

    if (!isa<ArrayAttr>(newFields)) {
      // Scalar result becomes hw.constant.
      IntegerAttr attr = cast<IntegerAttr>(newFields);
      auto result =
          hw::ConstantOp::create(rewriter, op.getLoc(), resultType, attr);
      rewriter.replaceOp(op, result);
      return success();
    }

    // Composite result becomes a new aggregate_constant op.
    auto newOp = hw::AggregateConstantOp::create(
        rewriter, op.getLoc(), resultType, cast<ArrayAttr>(newFields));
    rewriter.replaceOp(op, newOp);
    return success();
  }

  // NOLINTNEXTLINE(misc-no-recursion): Bounded recursion.
  Attribute rewriteArrayAttr(ArrayAttr array, Type type) const {
    if (getTypeConverter()->convertType(type) == type)
      return array;
    if (auto arrayType = dyn_cast<hw::ArrayType>(type);
        arrayType && arrayType.getNumElements() == 1) {
      return *array.begin();
    }

    // Collect the immediate subtypes. FieldIDTypeInterface is supported by
    // ArrayType, UnpackedArrayType, StructType, UnionType.
    auto fieldIdInterface = cast<hw::FieldIDTypeInterface>(type);
    SmallVector<Attribute> attrs;
    for (auto [index, attr] : llvm::enumerate(array)) {
      uint64_t fieldId = fieldIdInterface.getFieldID(index);
      Type subType = fieldIdInterface.getSubTypeByFieldID(fieldId).first;
      if (auto subArrayAttr = dyn_cast<ArrayAttr>(attr))
        attrs.push_back(rewriteArrayAttr(subArrayAttr, subType));
      else
        attrs.push_back(attr);
    }
    return ArrayAttr::get(array.getContext(), attrs);
  }
};

} // namespace

void RemoveI0TypesPass::runOnOperation() {
  TypeConverter converter;
  ConversionTarget target(getContext());
  RewritePatternSet patterns(&getContext());

  // The conversions for types are 1:N, where N may be 1 or 0.
  converter.addConversion([](Type type, SmallVectorImpl<Type> &types) {
    if (isI0(type))
      // Do not add any types to `types`.
      return success();
    types.push_back(type);
    return success();
  });

  // Composite types - recursively apply type conversion to inner types.
  converter.addConversion([&converter](hw::ArrayType type,
                                       SmallVectorImpl<Type> &types) {
    // If the array has only one element, we replace the array with the element.
    if (type.getNumElements() == 1) {
      Type converted = converter.convertType(type.getElementType());
      if (converted)
        types.push_back(converted);
      // The element was i0; no type is added to `types`.
      return success();
    }
    // Recursively apply type conversion to inner types.
    types.push_back(hw::ArrayType::get(
        converter.convertType(type.getElementType()), type.getNumElements()));
    return success();
  });
  converter.addConversion([&converter](hw::StructType type) -> Type {
    SmallVector<hw::StructType::FieldInfo> newMembers;
    // Convert and filter out any i0 fields.
    for (auto &field : type.getElements()) {
      SmallVector<Type> convertedTypes;
      if (failed(converter.convertType(field.type, convertedTypes)))
        return Type();
      if (!convertedTypes.empty()) {
        assert(convertedTypes.size() == 1);
        newMembers.push_back({field.name, convertedTypes[0]});
      }
    }
    return hw::StructType::get(type.getContext(), newMembers);
  });
  converter.addConversion([&converter](hw::UnionType type) -> Type {
    SmallVector<hw::UnionType::FieldInfo> newMembers;
    // Convert and filter out any i0 fields.
    for (auto &field : type.getElements()) {
      SmallVector<Type> convertedTypes;
      if (failed(converter.convertType(field.type, convertedTypes)))
        return Type();
      if (!convertedTypes.empty()) {
        assert(convertedTypes.size() == 1);
        newMembers.push_back({field.name, convertedTypes[0], field.offset});
      }
    }
    return hw::UnionType::get(type.getContext(), newMembers);
  });
  converter.addConversion(
      [&converter](hw::TypeAliasType type, SmallVectorImpl<Type> &types) {
        return converter.convertType(type.getCanonicalType(), types);
      });
  converter.addConversion(
      [&converter](arc::StateType type, SmallVectorImpl<Type> &types) {
        if (failed(converter.convertType(type.getType(), types)))
          return failure();
        assert(types.size() == 1);
        types[0] = arc::StateType::get(types[0]);
        return success();
      });

  target.markUnknownOpDynamicallyLegal(
      [&](Operation *op) { return converter.isLegal(op); });
  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp func) {
    FunctionType fty = func.getFunctionType();
    return converter.isLegal(fty.getInputs()) &&
           converter.isLegal(fty.getResults());
  });

  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 converter);

  patterns.add<LegalizeGeneric<func::ReturnOp>, LegalizeGeneric<func::CallOp>,
               LegalizeGeneric<hw::StructCreateOp>, ConvertArrayGet,
               ConvertArrayCreate, ConvertArrayInject, ConvertGeneric,
               ConvertAggregateConstant>(converter, &getContext());
  ConversionConfig config;
  config.allowPatternRollback = false;
  if (failed(applyFullConversion(getOperation(), target, std::move(patterns),
                                 config)))
    return signalPassFailure();
}
