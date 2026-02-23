//===- HWVectorization.cpp - HW Vectorization Pass --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass performs structural vectorization of hardware modules,
// merging scalar bit-level assignments into vectorized operations.
// This version handles linear, reverse, and mix vectorization using
// bit-tracking.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
namespace hw {
#define GEN_PASS_DEF_HWVECTORIZATION
#include "circt/Dialect/HW/Passes.h.inc"
} // namespace hw
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace comb;
using namespace hw;

namespace {

/// Represents a specific bit from a source SSA Value.
struct Bit {
  Value source;
  int index;

  Bit(Value source, int index) : source(source), index(index) {}
  Bit() : source(nullptr), index(0) {}
};

/// Maintains a mapping of bit indices to their source origins.
/// Uses SmallVector since the map is always dense (size == bitWidth).
struct BitArray {
  // Each element at position i holds the Bit for output bit i.
  // An unset entry has source == nullptr.
  llvm::SmallVector<Bit> bits;

  /// Checks if all bits form a linear sequence: output[i] <- source[i].
  bool isLinear(int size, Value sourceInput) const {
    if (bits.size() != static_cast<size_t>(size))
      return false;
    for (int i = 0; i < size; ++i) {
      if (bits[i].source != sourceInput || bits[i].index != i)
        return false;
    }
    return true;
  }

  /// Checks if all bits form a reverse sequence: output[i] <- source[N-1-i].
  bool isReverse(int size, Value sourceInput) const {
    if (bits.size() != static_cast<size_t>(size))
      return false;
    for (int i = 0; i < size; ++i) {
      if (bits[i].source != sourceInput || (size - 1) - i != bits[i].index)
        return false;
    }
    return true;
  }

  /// Returns the single source Value if all tracked bits share the same source.
  Value getSingleSourceValue() const {
    Value source = nullptr;
    for (const auto &bit : bits) {
      if (!bit.source)
        return nullptr;
      if (!source)
        source = bit.source;
      else if (source != bit.source)
        return nullptr;
    }
    return source;
  }

  size_t size() const { return bits.size(); }
};

class Vectorizer {
public:
  Vectorizer(hw::HWModuleOp module) : module(module) {}

  /// Analyzes bit-level provenance and applies vectorization transforms.
  void vectorize() {
    processOps();

    auto outputOp =
        dyn_cast<hw::OutputOp>(module.getBodyBlock()->getTerminator());
    if (!outputOp)
      return;

    IRRewriter rewriter(module.getContext());
    bool changed = false;

    for (Value oldOutputVal : outputOp->getOperands()) {
      auto type = dyn_cast<IntegerType>(oldOutputVal.getType());
      if (!type)
        continue;

      unsigned bitWidth = type.getWidth();
      auto it = bitArrays.find(oldOutputVal);
      if (it == bitArrays.end())
        continue;

      BitArray &arr = it->second;
      if (arr.size() != bitWidth)
        continue;

      Value sourceInput = arr.getSingleSourceValue();
      if (!sourceInput)
        continue;

      if (arr.isLinear(bitWidth, sourceInput)) {
        oldOutputVal.replaceAllUsesWith(sourceInput);
        changed = true;
      } else if (arr.isReverse(bitWidth, sourceInput)) {
        rewriter.setInsertionPointAfterValue(sourceInput);
        Value reversed = comb::ReverseOp::create(
            rewriter, sourceInput.getLoc(), sourceInput.getType(), sourceInput);
        oldOutputVal.replaceAllUsesWith(reversed);
        changed = true;
      } else if (isValidPermutation(arr, bitWidth)) {
        applyMixVectorization(rewriter, oldOutputVal, sourceInput, arr,
                              bitWidth);
        changed = true;
      }
    }

    if (changed)
      (void)mlir::runRegionDCE(rewriter, module.getBody());
  }

private:
  /// Maps values to their decomposed bit provenance.
  llvm::DenseMap<Value, BitArray> bitArrays;
  hw::HWModuleOp module;

  /// Checks that all bit indices are in [0, bitWidth] and form a bijection.
  /// Guards applyMixVectorization against malformed BitArrays.
  bool isValidPermutation(const BitArray &arr, unsigned bitWidth) {
    if (arr.size() != bitWidth)
      return false;
    llvm::SmallBitVector seen(bitWidth);
    for (const auto &bit : arr.bits) {
      assert(bit.index >= 0);
      if (bit.index >= static_cast<int>(bitWidth) || seen.test(bit.index))
        return false;
      seen.set(bit.index);
    }
    return true;
  }

  /// Handles arbitrary permutations from a single source by grouping runs of
  /// consecutive source-bit indices into ExtractOps, then concatenating them.
  ///
  /// Example: bits = [2, 3, 0, 1] produces:
  ///   %0 = extract src[2:1]   // bits 0-1 of output <- source[2:3]
  ///   %1 = extract src[0:1]   // bits 2-3 of output <- source[0:1]
  ///   %out = concat(%1, %0)   // MSB->LSB order
  void applyMixVectorization(IRRewriter &rewriter, Value oldOutputVal,
                             Value sourceInput, const BitArray &arr,
                             unsigned bitWidth) {
    rewriter.setInsertionPointAfterValue(sourceInput);
    Location loc = sourceInput.getLoc();

    // Walk output bits LSB->MSB, greedily extending each run while source
    // indices remain consecutive.
    llvm::SmallVector<Value> chunks;
    unsigned i = 0;
    while (i < bitWidth) {
      unsigned startBit = arr.bits[i].index;
      unsigned len = 1;
      while (i + len < bitWidth &&
             arr.bits[i + len].index == static_cast<int>(startBit + len))
        ++len;

      chunks.push_back(comb::ExtractOp::create(
          rewriter, loc, rewriter.getIntegerType(len), sourceInput, startBit));
      i += len;
    }

    // comb.concat expects operands MSB->LSB, so reverse the chunk list.
    std::reverse(chunks.begin(), chunks.end());

    Value newVal = comb::ConcatOp::create(
        rewriter, loc, rewriter.getIntegerType(bitWidth), chunks);

    oldOutputVal.replaceAllUsesWith(newVal);
  }

  /// Single walk that handles ExtractOp and ConcatOp using TypeSwitch.
  void processOps() {
    module.walk([&](Operation *op) {
      llvm::TypeSwitch<Operation *>(op)
          .Case<comb::ExtractOp>([&](comb::ExtractOp extractOp) {
            // Only handle single-bit extracts; skip multi-bit ranges.
            auto resultType =
                dyn_cast<IntegerType>(extractOp.getResult().getType());
            if (!resultType || resultType.getWidth() != 1)
              return;

            BitArray bits;
            bits.bits.push_back(
                Bit(extractOp.getInput(), extractOp.getLowBit()));
            bitArrays.insert({extractOp.getResult(), bits});
          })
          .Case<comb::ConcatOp>([&](comb::ConcatOp concatOp) {
            auto resultType =
                dyn_cast<IntegerType>(concatOp.getResult().getType());
            if (!resultType)
              return;

            unsigned totalWidth = resultType.getWidth();
            BitArray concatenatedArray;
            concatenatedArray.bits.resize(totalWidth);

            unsigned currentBitOffset = 0;
            for (Value operand : llvm::reverse(concatOp.getInputs())) {
              unsigned operandWidth =
                  cast<IntegerType>(operand.getType()).getWidth();
              auto it = bitArrays.find(operand);
              if (it != bitArrays.end()) {
                for (unsigned i = 0; i < it->second.bits.size(); ++i)
                  concatenatedArray.bits[i + currentBitOffset] =
                      it->second.bits[i];
              }
              currentBitOffset += operandWidth;
            }
            bitArrays.insert({concatOp.getResult(), concatenatedArray});
          });
    });
  }
};

struct HWVectorizationPass
    : public hw::impl::HWVectorizationBase<HWVectorizationPass> {

  void runOnOperation() override {
    Vectorizer v(getOperation());
    v.vectorize();
  }
};

} // namespace
