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

struct HWAggregateConstantOpConversion
    : OpConversionPattern<hw::AggregateConstantOp> {
  using OpConversionPattern<hw::AggregateConstantOp>::OpConversionPattern;

  static LogicalResult peelAttribute(Location loc, Attribute attr,
                                     ConversionPatternRewriter &rewriter,
                                     APInt &intVal) {
    SmallVector<Attribute> worklist;
    worklist.push_back(attr);
    unsigned nextInsertion = intVal.getBitWidth();

    while (!worklist.empty()) {
      auto current = worklist.pop_back_val();
      if (auto innerArray = dyn_cast<ArrayAttr>(current)) {
        for (auto elem : llvm::reverse(innerArray))
          worklist.push_back(elem);
        continue;
      }

      if (auto intAttr = dyn_cast<IntegerAttr>(current)) {
        auto chunk = intAttr.getValue();
        nextInsertion -= chunk.getBitWidth();
        intVal.insertBits(chunk, nextInsertion);
        continue;
      }

      return failure();
    }

    return success();
  }

  LogicalResult
  matchAndRewrite(hw::AggregateConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Lower to concat.
    SmallVector<Value> results;
    auto bitWidth = hw::getBitWidth(op.getType());
    assert(bitWidth >= 0 && "bit width must be known for constant");
    APInt intVal(bitWidth, 0);
    if (failed(peelAttribute(op.getLoc(), adaptor.getFieldsAttr(), rewriter,
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
    for (size_t i = 0; i < numElements; ++i)
      results.push_back(rewriter.createOrFold<comb::ExtractOp>(
          op.getLoc(), lowered, i * elemWidth, elemWidth));

    SmallVector<Value> bits;
    comb::extractBits(rewriter, op.getIndex(), bits);
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
      Value row = rewriter.create<hw::ArrayCreateOp>(loc, rowElements);
      arrayRows.push_back(row);
    }

    // Create the 2D array by concatenating all rows
    // arrayRows[0] corresponds to injection at index 0
    // arrayRows[1] corresponds to injection at index 1, etc.
    Value array2D = rewriter.create<hw::ArrayCreateOp>(loc, arrayRows);

    // Create array_get operation to select the row
    auto arrayGetOp =
        rewriter.create<hw::ArrayGetOp>(loc, array2D, adaptor.getIndex());

    rewriter.replaceOp(op, arrayGetOp);
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
      return IntegerType::get(t.getContext(), hw::getBitWidth(t));
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
  patterns.add<HWArrayGetOpConversion,
               HWArrayCreateLikeOpConversion<hw::ArrayCreateOp>,
               HWArrayCreateLikeOpConversion<hw::ArrayConcatOp>,
               HWAggregateConstantOpConversion, HWArrayInjectOpConversion>(
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

  // TODO: Add ArraySliceOp and struct operatons as well.
  target.addIllegalOp<hw::ArrayGetOp, hw::ArrayCreateOp, hw::ArrayConcatOp,
                      hw::AggregateConstantOp, hw::ArrayInjectOp>();

  target.addLegalDialect<hw::HWDialect, comb::CombDialect>();

  RewritePatternSet patterns(&getContext());
  AggregateTypeConverter typeConverter;
  populateHWAggregateToCombOpConversionPatterns(patterns, typeConverter);

  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
    return signalPassFailure();
}
