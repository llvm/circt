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
#include "circt/Dialect/LLHD/LLHDOps.h"
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

// An array_get with i0 index, or of a 1-element array (whose type gets
// scalarized regardless of the index width used), just returns the array
// input.
struct ConvertArrayGet : public OpConversionPattern<hw::ArrayGetOp> {
  using OpConversionPattern<hw::ArrayGetOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(hw::ArrayGetOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getIndex().empty() ||
        hw::type_cast<hw::ArrayType>(op.getInput().getType())
                .getNumElements() == 1) {
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

// Converts array_inject with i0 index, or into a 1-element array (whose type
// gets scalarized regardless of the index width used), to just return the
// element.
struct ConvertArrayInject : public OpConversionPattern<hw::ArrayInjectOp> {
  using OpConversionPattern<hw::ArrayInjectOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(hw::ArrayInjectOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getIndex().empty() ||
        hw::type_cast<hw::ArrayType>(op.getInput().getType())
                .getNumElements() == 1) {
      rewriter.replaceOp(op, adaptor.getElement());
      return success();
    }
    // Handled by ConvertGeneric.
    return failure();
  }
};

// A sig.array_get with an i0 index (the LLHD index-width constraint forces
// i0 exactly for 1-element arrays): the element ref aliases the whole-array
// ref (same address, [1 x T] layout). Ref TYPES are not scalarized (the
// signal's storage keeps its layout), so bridge the two ref types with an
// unrealized cast — the same bridging idiom the state lowering uses; both
// sides become plain pointers in the LLVM lowering and the cast cancels.
struct ConvertSigArrayGet : public OpConversionPattern<llhd::SigArrayGetOp> {
  using OpConversionPattern<llhd::SigArrayGetOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(llhd::SigArrayGetOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!adaptor.getIndex().empty())
      return failure();
    assert(adaptor.getInput().size() == 1);
    Value input = adaptor.getInput().front();
    if (input.getType() == op.getResult().getType()) {
      rewriter.replaceOp(op, input);
      return success();
    }
    auto castOp = mlir::UnrealizedConversionCastOp::create(
        rewriter, op.getLoc(), op.getResult().getType(), input);
    rewriter.replaceOp(op, castOp.getResults());
    return success();
  }
};

// The bit-slice sibling: an i0 low-bit index (1-bit signals) slices the
// whole width; the sliced ref aliases the input ref.
struct ConvertSigExtract : public OpConversionPattern<llhd::SigExtractOp> {
  using OpConversionPattern<llhd::SigExtractOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(llhd::SigExtractOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!adaptor.getLowBit().empty())
      return failure();
    assert(adaptor.getInput().size() == 1);
    Value input = adaptor.getInput().front();
    if (input.getType() == op.getResult().getType()) {
      rewriter.replaceOp(op, input);
      return success();
    }
    auto castOp = mlir::UnrealizedConversionCastOp::create(
        rewriter, op.getLoc(), op.getResult().getType(), input);
    rewriter.replaceOp(op, castOp.getResults());
    return success();
  }
};

// Probes and drives bridge the ref world (ref types keep the signal's
// storage layout and are not converted) to the value world (1-element
// arrays scalarized). When the value-side type converts, re-point the ref
// through an unrealized cast to the converted element type -- same address,
// a [1 x T] layout is byte-identical to T -- and rebuild the op.
struct ConvertProbe : public OpConversionPattern<llhd::ProbeOp> {
  using OpConversionPattern<llhd::ProbeOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(llhd::ProbeOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type oldType = op.getResult().getType();
    Type newType = getTypeConverter()->convertType(oldType);
    if (!newType || newType == oldType)
      return failure();
    assert(adaptor.getSignal().size() == 1);
    Value ref = mlir::UnrealizedConversionCastOp::create(
                    rewriter, op.getLoc(), llhd::RefType::get(newType),
                    adaptor.getSignal().front())
                    .getResult(0);
    rewriter.replaceOpWithNewOp<llhd::ProbeOp>(op, ref);
    return success();
  }
};

struct ConvertLLHDDrive : public OpConversionPattern<llhd::DriveOp> {
  using OpConversionPattern<llhd::DriveOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(llhd::DriveOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type oldType = op.getValue().getType();
    Type newType = getTypeConverter()->convertType(oldType);
    if (!newType || newType == oldType)
      return failure();
    assert(adaptor.getSignal().size() == 1 && adaptor.getValue().size() == 1 &&
           adaptor.getTime().size() == 1);
    Value ref = mlir::UnrealizedConversionCastOp::create(
                    rewriter, op.getLoc(), llhd::RefType::get(newType),
                    adaptor.getSignal().front())
                    .getResult(0);
    Value enable =
        adaptor.getEnable().empty() ? Value() : adaptor.getEnable().front();
    rewriter.replaceOpWithNewOp<llhd::DriveOp>(
        op, ref, adaptor.getValue().front(), adaptor.getTime().front(), enable);
    return success();
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
    if (!isI0(type))
      types.push_back(type);
    return success();
  });

  // Composite types - recursively apply type conversion to inner types.
  converter.addConversion([&converter](hw::ArrayType type,
                                       SmallVectorImpl<Type> &types) {
    // If the array has only one element, we replace the array with the element.
    if (type.getNumElements() == 1) {
      if (Type converted = converter.convertType(type.getElementType()))
        types.push_back(converted);
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
               ConvertArrayCreate, ConvertArrayInject, ConvertSigArrayGet,
               ConvertSigExtract, ConvertProbe, ConvertLLHDDrive,
               ConvertGeneric, ConvertAggregateConstant>(converter,
                                                         &getContext());
  ConversionConfig config;
  config.allowPatternRollback = false;
  if (failed(applyFullConversion(getOperation(), target, std::move(patterns),
                                 config)))
    return signalPassFailure();
}
