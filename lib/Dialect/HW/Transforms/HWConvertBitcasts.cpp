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
  // If input is defined by BitcastOp, unwrap it:
  if (auto bitcast = inputVal.getDefiningOp<BitcastOp>();
      bitcast &&
      HWConvertBitcastsPass::isTypeSupported(bitcast.getInput().getType())) {
    collectIntegersRecursively(builder, loc, bitcast.getInput(), accumulator);
    return;
  }

  // If input is defined by ConcatOp, unwrap its operands:
  if (auto concat = inputVal.getDefiningOp<comb::ConcatOp>()) {
    for (Value op : concat.getInputs())
      collectIntegersRecursively(builder, loc, op, accumulator);
    return;
  }

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
    // Avoid creating zero-width indices to dodge lowering issues
    auto indexType =
        builder.getIntegerType(std::max(1u, llvm::Log2_64_Ceil(numElements)));
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

// Convert collected integer chunks (MSB to LSB) to aggregate
// NOLINTNEXTLINE(misc-no-recursion)
static Value constructAggregateRecursively(OpBuilder builder, Location loc,
                                           ArrayRef<Value> integers,
                                           size_t &idx, size_t &bitOffset,
                                           Type targetType) {
  assert(getBitWidth(targetType) >= 0 && "Bitwidth of target must be known");

  // End of recursion: Integer value
  if (auto intTy = dyn_cast<IntegerType>(targetType)) {
    size_t remaining = intTy.getWidth();
    if (remaining == 0)
      return ConstantOp::create(builder, loc, builder.getIntegerAttr(intTy, 0));

    SmallVector<Value> pieces;
    while (remaining > 0) {
      assert(idx < integers.size() && "Not enough bits in input");
      Value cur = integers[idx];
      size_t curWidth = cur.getType().getIntOrFloatBitWidth();
      if (curWidth == 0) {
        ++idx;
        continue;
      }
      size_t take = std::min(remaining, curWidth - bitOffset);
      if (bitOffset == 0 && take == curWidth) {
        pieces.push_back(cur);
      } else {
        auto sliceTy = builder.getIntegerType(take);
        pieces.push_back(comb::ExtractOp::create(builder, loc, sliceTy, cur,
                                                 curWidth - bitOffset - take));
      }
      remaining -= take;
      if ((bitOffset += take) == curWidth) {
        ++idx;
        bitOffset = 0;
      }
    }
    if (pieces.size() == 1)
      return pieces.front();
    return comb::ConcatOp::create(builder, loc, pieces);
  }

  SmallVector<Value> elements;

  // Array Type
  if (auto arrayTy = type_dyn_cast<ArrayType>(targetType)) {
    auto numElements = arrayTy.getNumElements();
    elements.reserve(numElements);
    for (unsigned i = 0; i < numElements; ++i) {
      // Process bits from MSB to LSB
      elements.push_back(constructAggregateRecursively(
          builder, loc, integers, idx, bitOffset, arrayTy.getElementType()));
    }
    // Array create reverses order:
    // Higher bits are added first and become high indices.
    return ArrayCreateOp::create(builder, loc, targetType, elements);
  }

  // Struct Type
  if (auto structTy = type_dyn_cast<StructType>(targetType)) {
    auto fields = structTy.getElements();
    elements.reserve(fields.size());
    for (const auto &field : fields) {
      // Process bits from MSB to LSB
      elements.push_back(constructAggregateRecursively(
          builder, loc, integers, idx, bitOffset, field.type));
    }
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

  // Collect integer chunks from input value (MSB to LSB)
  SmallVector<Value> integers;
  collectIntegersRecursively(builder, bitcastOp.getLoc(), bitcastOp.getInput(),
                             integers);

  // Convert collected integer chunks to the target type
  size_t idx = 0, bitOffset = 0;
  Value result =
      constructAggregateRecursively(builder, bitcastOp.getLoc(), integers, idx,
                                    bitOffset, bitcastOp.getType());

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
