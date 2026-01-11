//===- ConvertBitcasts.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "hw-convert-bitcasts"

namespace circt {
namespace hw {
#define GEN_PASS_DEF_HWCONVERTBITCASTS
#include "circt/Dialect/HW/Passes.h.inc"
} // namespace hw
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace hw;

namespace {
struct HWConvertBitcastsPass
    : circt::hw::impl::HWConvertBitcastsBase<HWConvertBitcastsPass> {
  using circt::hw::impl::HWConvertBitcastsBase<
      HWConvertBitcastsPass>::HWConvertBitcastsBase;

  void runOnOperation() override;
  static bool isTypeSupported(Type ty);
  LogicalResult convertBitcastOp(OpBuilder builder, BitcastOp bitcastOp);
};
} // namespace

// Array conversion:  Lower bits correspond to lower array index
// Struct conversion: Higher bits correspond to lower field index

// NOLINTNEXTLINE(misc-no-recursion)
bool HWConvertBitcastsPass::isTypeSupported(Type ty) {
  if (isa<IntegerType>(ty))
    return true;
  if (auto arrayTy = hw::type_dyn_cast<hw::ArrayType>(ty))
    return isTypeSupported(arrayTy.getElementType());
  if (auto structTy = hw::type_dyn_cast<hw::StructType>(ty))
    return llvm::all_of(structTy.getElements(),
                        [](StructType::FieldInfo field) {
                          return isTypeSupported(field.type);
                        });
  // TODO: Add support for: union, packed array, enum
  return false;
}

// Collect integer fields of aggregates for concatenation
// NOLINTNEXTLINE(misc-no-recursion)
static void collectIntegersRecursively(OpBuilder builder, Location loc,
                                       Value inputVal,
                                       SmallVectorImpl<Value> &accumulator) {
  // End of recursion: Integer value
  if (isa<IntegerType>(inputVal.getType())) {
    accumulator.push_back(inputVal);
    return;
  }

  auto numBits = getBitWidth(inputVal.getType());
  assert(numBits >= 0 && "Bitwidth of input must be known");

  // Array Type
  if (auto arrayTy = dyn_cast<ArrayType>(inputVal.getType())) {
    unsigned numElements = arrayTy.getNumElements();
    auto indexType = builder.getIntegerType(llvm::Log2_64_Ceil(numElements));
    for (unsigned i = 0; i < numElements; ++i) {
      // Process from high to low array index
      auto indexCst = ConstantOp::create(
          builder, loc, builder.getIntegerAttr(indexType, numElements - i - 1));
      auto getOp = ArrayGetOp::create(builder, loc, inputVal, indexCst);
      collectIntegersRecursively(builder, loc, getOp.getResult(), accumulator);
    }
    return;
  }

  // Struct Type
  if (auto structTy = dyn_cast<StructType>(inputVal.getType())) {
    // Process from low to high field index
    auto explodeOp = StructExplodeOp::create(builder, loc, inputVal);
    for (auto elt : explodeOp.getResults())
      collectIntegersRecursively(builder, loc, elt, accumulator);
    return;
  }

  assert(false && "Unsupported type");
}

// Convert integer to aggregate
// NOLINTNEXTLINE(misc-no-recursion)
static Value constructAggregateRecursively(OpBuilder builder, Location loc,
                                           Value rawInteger, Type targetType) {
  auto numBits = getBitWidth(targetType);
  assert(numBits >= 0 && "Bitwidth of target must be known");
  assert(numBits == rawInteger.getType().getIntOrFloatBitWidth());

  // End of recursion: Integer value
  if (isa<IntegerType>(targetType))
    return rawInteger;

  SmallVector<Value> elements;

  // Array Type
  if (auto arrayTy = type_dyn_cast<ArrayType>(targetType)) {
    auto numElements = arrayTy.getNumElements();
    auto sliceWidth = getBitWidth(arrayTy.getElementType());
    assert(sliceWidth >= 0);
    auto sliceTy = builder.getIntegerType(sliceWidth);
    elements.reserve(numElements);
    for (unsigned i = 0; i < numElements; ++i) {
      // Process bits from MSB to LSB
      auto offset = sliceWidth * (numElements - i - 1);
      Value slice;
      if (sliceWidth == 0)
        slice = ConstantOp::create(builder, loc,
                                   builder.getIntegerAttr(sliceTy, 0));
      else
        slice =
            comb::ExtractOp::create(builder, loc, sliceTy, rawInteger, offset);
      auto elt = constructAggregateRecursively(builder, loc, slice,
                                               arrayTy.getElementType());
      elements.push_back(elt);
    }
    // Array create reverses order:
    // Higher bits are added first and become high indices.
    return ArrayCreateOp::create(builder, loc, targetType, elements);
  }

  // Struct Type
  if (auto structTy = type_dyn_cast<StructType>(targetType)) {
    auto numElements = structTy.getElements().size();
    unsigned consumedBits = 0;
    for (unsigned i = 0; i < numElements; ++i) {

      auto eltBits = getBitWidth(structTy.getElements()[i].type);
      assert(eltBits >= 0);
      Value slice;
      // Process bits from MSB to LSB
      if (eltBits == 0)
        slice = ConstantOp::create(
            builder, loc, builder.getIntegerAttr(builder.getIntegerType(0), 0));
      else
        slice = comb::ExtractOp::create(
            builder, loc, builder.getIntegerType(eltBits), rawInteger,
            numBits - consumedBits - eltBits);
      auto elt = constructAggregateRecursively(builder, loc, slice,
                                               structTy.getElements()[i].type);
      elements.push_back(elt);
      consumedBits += eltBits;
    }
    assert(consumedBits == numBits);
    // Struct create does not reverse order:
    // Higher bits are added first and become low indices.
    return StructCreateOp::create(builder, loc, targetType, elements);
  }

  assert(false && "Unsupported type");
  return {};
}

LogicalResult HWConvertBitcastsPass::convertBitcastOp(OpBuilder builder,
                                                      BitcastOp bitcastOp) {
  bool inputSupported = isTypeSupported(bitcastOp.getInput().getType());
  bool outputSupported = isTypeSupported(bitcastOp.getType());
  if (!allowPartialConversion) {
    if (!inputSupported)
      bitcastOp.emitOpError("has unsupported input type");
    if (!outputSupported)
      bitcastOp.emitOpError("has unsupported output type");
  }
  if (!(inputSupported && outputSupported))
    return failure();

  builder.setInsertionPoint(bitcastOp);

  // Convert input value to a packed integer
  SmallVector<Value> integers;
  collectIntegersRecursively(builder, bitcastOp.getLoc(), bitcastOp.getInput(),
                             integers);
  Value concat;
  if (integers.size() == 1)
    concat = integers.front();
  else
    concat = comb::ConcatOp::create(builder, bitcastOp.getLoc(), integers)
                 .getResult();

  // Convert packed integer to the target type
  auto result = constructAggregateRecursively(builder, bitcastOp.getLoc(),
                                              concat, bitcastOp.getType());

  // Replace operation
  bitcastOp.getResult().replaceAllUsesWith(result);
  bitcastOp.erase();
  return success();
}

void HWConvertBitcastsPass::runOnOperation() {
  OpBuilder builder(getOperation());
  bool anyFailed = false;
  bool anyChanged = false;
  getOperation().getBody()->walk<WalkOrder::PostOrder, ReverseIterator>(
      [&](BitcastOp bitcastOp) {
        anyChanged = true;
        if (failed(convertBitcastOp(builder, bitcastOp)))
          anyFailed = true;
      });

  if (!anyChanged) {
    markAllAnalysesPreserved();
    return;
  }

  if (anyFailed && !allowPartialConversion)
    signalPassFailure();
}
