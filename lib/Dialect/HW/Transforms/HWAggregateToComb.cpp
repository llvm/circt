//===- HWAggregateToComb.cpp - HW aggregate to comb -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APInt.h"

namespace circt {
namespace hw {
#define GEN_PASS_DEF_HWAGGREGATETOCOMB
#include "circt/Dialect/HW/Passes.h.inc"
} // namespace hw
} // namespace circt

using namespace mlir;
using namespace circt;

namespace {

// Lower hw.array_create and hw.array_concat to comb.concat.
template <typename OpTy>
struct HWArrayCreateLikeOpConversion : OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<OpTy>::OpAdaptor;
  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<comb::ConcatOp>(op, adaptor.getInputs());
    return success();
  }
};

struct HWUnionCreateOpConversion
    : public OpConversionPattern<hw::UnionCreateOp> {
  using OpConversionPattern<hw::UnionCreateOp>::OpConversionPattern;
  // hw.union_create -> hw.bitcast [ + comb.concat ]
  LogicalResult
  matchAndRewrite(hw::UnionCreateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    hw::UnionType unionTy = op.getType();
    auto outputTy =
        dyn_cast_or_null<IntegerType>(typeConverter->convertType(unionTy));
    if (!outputTy)
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "Failed to convert union to integer");

    auto inputBitWidth = hw::getBitWidth(adaptor.getInput().getType());
    if (inputBitWidth < 0)
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "Failed to convert input to integer");

    // Bitcast the input value to its integer representation.
    auto inputIntTy = rewriter.getIntegerType(inputBitWidth);
    Value inputAsInt = rewriter.createOrFold<hw::BitcastOp>(
        op.getLoc(), inputIntTy, adaptor.getInput());

    // The field shares the LSB of the union and is moved towards the MSB by
    // its offset. The bits the field does not cover are undefined and filled
    // with zeros.
    int64_t bitOffset = unionTy.getElements()[op.getFieldIndex()].offset;
    int64_t prePadding = outputTy.getWidth() - inputBitWidth - bitOffset;

    auto createZeroCst = [&](Location loc, int64_t bitWidth) -> Value {
      return hw::ConstantOp::create(rewriter, loc,
                                    rewriter.getIntegerType(bitWidth), 0);
    };

    SmallVector<Value> concatOperands;
    if (prePadding > 0)
      concatOperands.push_back(createZeroCst(op.getLoc(), prePadding));
    concatOperands.push_back(inputAsInt);
    if (bitOffset > 0)
      concatOperands.push_back(createZeroCst(op.getLoc(), bitOffset));

    Value result =
        rewriter.createOrFold<comb::ConcatOp>(op.getLoc(), concatOperands);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct HWUnionExtractOpConversion
    : public OpConversionPattern<hw::UnionExtractOp> {
  using OpConversionPattern<hw::UnionExtractOp>::OpConversionPattern;
  // hw.union_extract -> [ comb.extract + ] hw.bitcast
  LogicalResult
  matchAndRewrite(hw::UnionExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    hw::UnionType unionTy = op.getInput().getType();

    auto inputTy = dyn_cast_or_null<IntegerType>(adaptor.getInput().getType());
    if (!inputTy)
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "Failed to convert union to integer");
    auto outputTy = typeConverter->convertType(op.getType());
    if (!outputTy)
      return rewriter.notifyMatchFailure(
          op.getLoc(), "Failed to convert union extract result type");

    auto resultFieldBits = hw::getBitWidth(outputTy);
    assert(resultFieldBits >= 0);
    auto integerValue = adaptor.getInput();

    // If the output is narrower than the union, extract the active bits.
    if (resultFieldBits < integerValue.getType().getIntOrFloatBitWidth()) {
      auto bitOffset = unionTy.getElements()[op.getFieldIndex()].offset;
      integerValue = comb::ExtractOp::create(
          rewriter, op->getLoc(), rewriter.getIntegerType(resultFieldBits),
          integerValue, bitOffset);
    }

    // Bitcast the extracted bits to the result. Fold inplace if outputTy ==
    // inputTy.
    auto bitcastOp = rewriter.createOrFold<hw::BitcastOp>(op.getLoc(), outputTy,
                                                          integerValue);

    rewriter.replaceOp(op, bitcastOp);
    return success();
  }
};

struct HWAggregateConstantOpConversion
    : OpConversionPattern<hw::AggregateConstantOp> {
  using OpConversionPattern<hw::AggregateConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hw::AggregateConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Lower to concat.
    APInt intVal;
    if (failed(hw::aggregateAttrToAPInt(op.getType(), adaptor.getFieldsAttr(),
                                        intVal)))
      return failure();
    rewriter.replaceOpWithNewOp<hw::ConstantOp>(op, intVal);
    return success();
  }
};

struct HWArrayGetOpConversion : OpConversionPattern<hw::ArrayGetOp> {
  using OpConversionPattern<hw::ArrayGetOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hw::ArrayGetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> results;
    auto arrayType = cast<hw::ArrayType>(op.getInput().getType());
    auto elemType = arrayType.getElementType();
    auto numElements = arrayType.getNumElements();
    auto elemWidth = hw::getBitWidth(elemType);
    if (elemWidth < 0)
      return rewriter.notifyMatchFailure(op.getLoc(), "unknown element width");

    auto lowered = adaptor.getInput();
    auto index = adaptor.getIndex();
    APInt constantIndex;
    if (matchPattern(index, m_ConstantInt(&constantIndex))) {
      int64_t maxIndex = std::numeric_limits<int32_t>::max() / elemWidth;
      if (constantIndex.isSingleWord() &&
          constantIndex.getZExtValue() <= static_cast<uint64_t>(maxIndex)) {
        rewriter.replaceOpWithNewOp<comb::ExtractOp>(
            op, lowered, constantIndex.getZExtValue() * elemWidth, elemWidth);
        return success();
      }
    }

    for (size_t i = 0; i < numElements; ++i)
      results.push_back(rewriter.createOrFold<comb::ExtractOp>(
          op.getLoc(), lowered, i * elemWidth, elemWidth));

    SmallVector<Value> bits;
    comb::extractBits(rewriter, index, bits);
    auto result = comb::constructMuxTree(rewriter, op.getLoc(), bits, results,
                                         results.back());

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct HWArraySliceOpConversion : OpConversionPattern<hw::ArraySliceOp> {
  using OpConversionPattern<hw::ArraySliceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hw::ArraySliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> results;
    auto arrayType = cast<hw::ArrayType>(op.getInput().getType());
    auto elemType = arrayType.getElementType();
    auto numElements = arrayType.getNumElements();
    auto elemWidth = hw::getBitWidth(elemType);
    if (elemWidth < 0)
      return rewriter.notifyMatchFailure(op.getLoc(), "unknown element width");
    auto resultArrayType = cast<hw::ArrayType>(op.getResult().getType());
    auto resultNumElements = resultArrayType.getNumElements();

    auto lowered = adaptor.getInput();
    auto index = adaptor.getLowIndex();
    APInt constantIndex;
    if (matchPattern(index, m_ConstantInt(&constantIndex))) {
      int64_t maxIndex = std::numeric_limits<int32_t>::max() / elemWidth;
      if (constantIndex.isSingleWord() &&
          constantIndex.getZExtValue() <= static_cast<uint64_t>(maxIndex)) {
        rewriter.replaceOpWithNewOp<comb::ExtractOp>(
            op, lowered, constantIndex.getZExtValue() * elemWidth,
            resultNumElements * elemWidth);
        return success();
      }
    }

    for (size_t i = 0; i <= numElements - resultNumElements; ++i)
      results.push_back(rewriter.createOrFold<comb::ExtractOp>(
          op.getLoc(), lowered, i * elemWidth, resultNumElements * elemWidth));

    SmallVector<Value> bits;
    comb::extractBits(rewriter, index, bits);
    auto result = comb::constructMuxTree(rewriter, op.getLoc(), bits, results,
                                         results.back());

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct HWArrayInjectOpConversion : OpConversionPattern<hw::ArrayInjectOp> {
  using OpConversionPattern<hw::ArrayInjectOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hw::ArrayInjectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto arrayType = cast<hw::ArrayType>(op.getInput().getType());
    auto elemType = arrayType.getElementType();
    auto numElements = arrayType.getNumElements();
    auto elemWidth = hw::getBitWidth(elemType);
    if (elemWidth < 0)
      return rewriter.notifyMatchFailure(op.getLoc(), "unknown element width");

    Location loc = op.getLoc();

    // Extract all elements from the input array
    SmallVector<Value> originalElements;
    auto inputArray = adaptor.getInput();
    for (size_t i = 0; i < numElements; ++i) {
      originalElements.push_back(rewriter.createOrFold<comb::ExtractOp>(
          loc, inputArray, i * elemWidth, elemWidth));
    }

    // Create 2D array: each row represents what the array would look like
    // if injection happened at that specific index
    SmallVector<Value> arrayRows;
    arrayRows.reserve(numElements);
    for (int injectIdx = numElements - 1; injectIdx >= 0; --injectIdx) {
      SmallVector<Value> rowElements;
      rowElements.reserve(numElements);

      // Build the row: array[n-1], array[n-2], ..., but replace element at
      // injectIdx with newVal
      for (int originalIdx = numElements - 1; originalIdx >= 0; --originalIdx) {
        if (originalIdx == injectIdx) {
          rowElements.push_back(adaptor.getElement());
        } else {
          rowElements.push_back(originalElements[originalIdx]);
        }
      }

      // Concatenate elements to form this row
      Value row = hw::ArrayCreateOp::create(rewriter, loc, rowElements);
      arrayRows.push_back(row);
    }

    // Create the 2D array by concatenating all rows
    // arrayRows[0] corresponds to injection at index 0
    // arrayRows[1] corresponds to injection at index 1, etc.
    Value array2D = hw::ArrayCreateOp::create(rewriter, loc, arrayRows);

    // Create array_get operation to select the row
    auto arrayGetOp =
        hw::ArrayGetOp::create(rewriter, loc, array2D, adaptor.getIndex());

    rewriter.replaceOp(op, arrayGetOp);
    return success();
  }
};

struct HWStructCreateOpConversion : OpConversionPattern<hw::StructCreateOp> {
  using OpConversionPattern<hw::StructCreateOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hw::StructCreateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Lower struct_create to comb.concat. The first field occupies the MSBs, so
    // we concatenate fields in order (comb.concat places first operand at MSB).
    rewriter.replaceOpWithNewOp<comb::ConcatOp>(op, adaptor.getInput());
    return success();
  }
};

struct HWStructExtractOpConversion : OpConversionPattern<hw::StructExtractOp> {
  using OpConversionPattern<hw::StructExtractOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hw::StructExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto structType = cast<hw::StructType>(op.getInput().getType());
    auto fieldIndex = op.getFieldIndex();
    auto elements = structType.getElements();

    int64_t totalBitWidth = hw::getBitWidth(structType);
    if (totalBitWidth < 0)
      return rewriter.notifyMatchFailure(op.getLoc(), "unknown struct width");

    // Compute the bit offset from the MSB by summing the widths of all
    // preceding fields. The first field occupies the MSBs.
    int64_t consumedBits = 0;
    for (size_t i = 0; i < fieldIndex; ++i) {
      int64_t fieldWidth = hw::getBitWidth(elements[i].type);
      assert(fieldWidth >= 0 &&
             "must be failed before if field width is unknown");
      consumedBits += fieldWidth;
    }

    int64_t fieldWidth = hw::getBitWidth(elements[fieldIndex].type);
    assert(fieldWidth >= 0 &&
           "must be failed before if field width is unknown");

    // Extract the field using comb.extract. Offset is from LSB.
    int64_t bitOffset = totalBitWidth - consumedBits - fieldWidth;
    rewriter.replaceOpWithNewOp<comb::ExtractOp>(op, adaptor.getInput(),
                                                 bitOffset, fieldWidth);
    return success();
  }
};

struct BitcastOpConversion : OpConversionPattern<hw::BitcastOp> {
  using OpConversionPattern<hw::BitcastOp>::OpConversionPattern;
  // Recreate bitcast with legalized types.
  LogicalResult
  matchAndRewrite(hw::BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto inputTy = adaptor.getInput().getType();
    auto outputTy = typeConverter->convertType(op.getType());
    if (!outputTy)
      return rewriter.notifyMatchFailure(op, "Failed to convert result type.");

    auto inBits = hw::getBitWidth(inputTy);
    auto outBits = hw::getBitWidth(outputTy);
    if (inBits != outBits)
      return rewriter.notifyMatchFailure(
          op, "Width of converted types does not match.");
    if (inBits < 0)
      return rewriter.notifyMatchFailure(op, "Unknown bitwidth.");

    auto bitcastOp = rewriter.createOrFold<hw::BitcastOp>(op.getLoc(), outputTy,
                                                          adaptor.getInput());
    rewriter.replaceOp(op, bitcastOp);
    return success();
  }
};

struct MuxOpConversion : OpConversionPattern<comb::MuxOp> {
  using OpConversionPattern<comb::MuxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comb::MuxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Re-create Mux with legalized types.
    rewriter.replaceOpWithNewOp<comb::MuxOp>(
        op, adaptor.getCond(), adaptor.getTrueValue(), adaptor.getFalseValue());
    return success();
  }
};

/// A type converter is needed to perform the in-flight materialization of
/// aggregate types to integer types.
class AggregateTypeConverter : public TypeConverter {
public:
  AggregateTypeConverter() {
    addConversion([](Type type) -> Type { return type; });
    addConversion([](hw::ArrayType t) -> Type {
      auto bitWidth = t.getBitWidth();
      if (!bitWidth)
        return {};
      return IntegerType::get(t.getContext(), *bitWidth);
    });
    addConversion([](hw::StructType t) -> Type {
      auto bitWidth = t.getBitWidth();
      if (!bitWidth)
        return {};
      return IntegerType::get(t.getContext(), *bitWidth);
    });
    addConversion([](hw::UnionType t) -> Type {
      auto bitWidth = t.getBitWidth();
      if (!bitWidth)
        return {};
      return IntegerType::get(t.getContext(), *bitWidth);
    });
    addTargetMaterialization([](mlir::OpBuilder &builder, mlir::Type resultType,
                                mlir::ValueRange inputs,
                                mlir::Location loc) -> mlir::Value {
      if (inputs.size() != 1)
        return Value();

      return hw::BitcastOp::create(builder, loc, resultType, inputs[0])
          ->getResult(0);
    });

    addSourceMaterialization([](mlir::OpBuilder &builder, mlir::Type resultType,
                                mlir::ValueRange inputs,
                                mlir::Location loc) -> mlir::Value {
      if (inputs.size() != 1)
        return Value();

      return hw::BitcastOp::create(builder, loc, resultType, inputs[0])
          ->getResult(0);
    });
  }
};
} // namespace

static void populateHWAggregateToCombOpConversionPatterns(
    RewritePatternSet &patterns, AggregateTypeConverter &typeConverter) {
  patterns.add<
      HWArrayGetOpConversion, HWArrayCreateLikeOpConversion<hw::ArrayCreateOp>,
      HWArrayCreateLikeOpConversion<hw::ArrayConcatOp>,
      HWAggregateConstantOpConversion, HWArraySliceOpConversion,
      HWArrayInjectOpConversion, HWStructCreateOpConversion,
      HWStructExtractOpConversion, HWUnionCreateOpConversion,
      HWUnionExtractOpConversion, BitcastOpConversion, MuxOpConversion>(
      typeConverter, patterns.getContext());
}

namespace {
struct HWAggregateToCombPass
    : public hw::impl::HWAggregateToCombBase<HWAggregateToCombPass> {
  void runOnOperation() override;
  using HWAggregateToCombBase<HWAggregateToCombPass>::HWAggregateToCombBase;
};
} // namespace

void HWAggregateToCombPass::runOnOperation() {
  ConversionTarget target(getContext());

  target.addIllegalOp<hw::ArrayGetOp, hw::ArrayCreateOp, hw::ArrayConcatOp,
                      hw::AggregateConstantOp, hw::ArrayInjectOp,
                      hw::ArraySliceOp, hw::StructCreateOp, hw::StructExtractOp,
                      hw::UnionCreateOp, hw::UnionExtractOp>();
  target.addLegalDialect<hw::HWDialect, comb::CombDialect>();

  RewritePatternSet patterns(&getContext());
  AggregateTypeConverter typeConverter;
  populateHWAggregateToCombOpConversionPatterns(patterns, typeConverter);

  target.addDynamicallyLegalOp<comb::MuxOp, hw::BitcastOp>(
      [&typeConverter](auto op) { return typeConverter.isLegal(op); });

  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
    return signalPassFailure();
}
