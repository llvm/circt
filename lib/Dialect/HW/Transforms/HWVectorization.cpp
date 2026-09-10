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
// This version handles linear, reverse, and mix permutation and
// structural vectorization using bit-tracking.
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

  /// Entry point: analyze provenance then apply the best vectorization for
  /// each output port.
  void vectorize() {
    // Phase 1: populate `bitArrays` by walking all ops in program order.
    processOps();

    auto outputOp =
        dyn_cast<hw::OutputOp>(module.getBodyBlock()->getTerminator());
    if (!outputOp)
      return;

    IRRewriter rewriter(module.getContext());
    bool changed = false;

    // Phase 2: for each integer output, attempt vectorization strategies in
    // order of increasing complexity:
    //   (a) Linear   – direct wire-through  -> drop the concat entirely
    //   (b) Reverse  – mirror permutation   -> comb.reverse
    //   (c) Mix      – arbitrary bijection  -> extract + concat chunks
    //   (d) Structural – isomorphic scalar cones -> wide AND/OR/XOR/MUX
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

      // `transformed` tracks whether *this* output was successfully vectorized.
      // It must be local to each iteration so that a successful transform on
      // one output port does not suppress strategy (d) for a later port.
      bool transformed = false;

      // 1. Try vectorizing from a single source (Linear, Reverse, Mix).
      if (sourceInput) {
        if (arr.isLinear(bitWidth, sourceInput)) {
          oldOutputVal.replaceAllUsesWith(sourceInput);
          transformed = true;
        } else if (arr.isReverse(bitWidth, sourceInput)) {
          rewriter.setInsertionPointAfterValue(sourceInput);
          Value reversed =
              comb::ReverseOp::create(rewriter, sourceInput.getLoc(),
                                      sourceInput.getType(), sourceInput);
          oldOutputVal.replaceAllUsesWith(reversed);
          transformed = true;
        } else if (isValidPermutation(arr, bitWidth)) {
          applyMixVectorization(rewriter, oldOutputVal, sourceInput, arr,
                                bitWidth);
          transformed = true;
        }
      }

      // 2. If it wasn't vectorized (or if it has multiple sources), try
      // Structural.
      if (!transformed && !hasCrossBitDependencies(oldOutputVal) &&
          canVectorizeStructurally(oldOutputVal)) {
        rewriter.setInsertionPointAfterValue(oldOutputVal);

        unsigned width = cast<IntegerType>(oldOutputVal.getType()).getWidth();
        Value slice0 = findBitSource(oldOutputVal, 0);
        if (slice0) {
          DenseMap<Value, Value> vectorizedMap;
          Value vec = vectorizeSubgraph(rewriter, slice0, width, vectorizedMap);
          if (vec) {
            oldOutputVal.replaceAllUsesWith(vec);
            transformed = true;
          }
        }
      }

      if (transformed)
        changed = true;
    }

    if (changed)
      (void)mlir::runRegionDCE(rewriter, module.getBody());
  }

private:
  /// Maps each SSA Value to its bit-level provenance after the analysis phase.
  llvm::DenseMap<Value, BitArray> bitArrays;
  hw::HWModuleOp module;

  /// Analyzes the logic cones of all bit lanes to detect illegal cross-bit
  /// dependencies in O(bitWidth + N) time.
  bool hasCrossBitDependencies(mlir::Value outputVal) {
    unsigned bitWidth = cast<IntegerType>(outputVal.getType()).getWidth();

    llvm::DenseSet<mlir::Value> visitedUnsafe; // Accumulate unsafe values.
    llvm::SmallVector<mlir::Value> worklist;

    for (unsigned i = 0; i < bitWidth; ++i) {
      llvm::DenseSet<mlir::Value> visitedLocal;
      mlir::Value bitSource = findBitSource(outputVal, i);
      if (!bitSource)
        continue;

      worklist.push_back(bitSource);
      while (!worklist.empty()) {
        auto top = worklist.pop_back_val();
        if (isSafeSharedValue(top))
          continue; // don't add to the set.
        if (!visitedLocal.insert(top).second)
          continue; // Arriving multiple time in the same iteration is fine.
        // If it's already visited, there is a depencency
        if (!visitedUnsafe.insert(top).second)
          return true;
        if (auto *op = top.getDefiningOp()) {
          for (auto operand : op->getOperands())
            worklist.push_back(operand);
        }
      }
    }
    return false;
  }

  /// Determines if a shared value is safe for vectorization. Only constants
  /// and block arguments are safe to share between bit lanes. Any intermediate
  /// operation is considered unsafe as it may introduce cross-lane
  /// dependencies.
  bool isSafeSharedValue(mlir::Value val) {
    return val &&
           (isa<BlockArgument>(val) || val.getDefiningOp<hw::ConstantOp>());
  }

  /// Checks if a logic cone is composed of structurally equivalent slices
  /// that can be merged into a vector operation.
  ///
  /// The check succeeds when every bit slice i of the output is produced by a
  /// subgraph that is isomorphic to the bit-0 subgraph (slice0).
  bool canVectorizeStructurally(Value output) {
    unsigned bitWidth = cast<IntegerType>(output.getType()).getWidth();
    if (bitWidth <= 1)
      return false;

    Value slice0Val = findBitSource(output, 0);
    if (!slice0Val)
      return false;

    Value slice1Val = findBitSource(output, 1);
    if (!slice1Val)
      return false;

    auto extract0 = slice0Val.getDefiningOp<comb::ExtractOp>();
    auto extract1 = slice1Val.getDefiningOp<comb::ExtractOp>();

    if (!extract0 || !extract1 || extract0.getInput() != extract1.getInput()) {
      for (unsigned i = 1; i < bitWidth; ++i) {
        Value sliceNVal = findBitSource(output, i);
        if (!sliceNVal || !sliceNVal.getDefiningOp())
          return false;
        llvm::DenseMap<mlir::Value, mlir::Value> map;
        if (!areSubgraphsEquivalent(slice0Val, sliceNVal, i, 1, map)) {
          return false;
        }
      }
      return true;
    }

    int stride = (int)extract1.getLowBit() - (int)extract0.getLowBit();

    for (unsigned i = 1; i < bitWidth; ++i) {
      Value sliceNVal = findBitSource(output, i);
      if (!sliceNVal || !sliceNVal.getDefiningOp())
        return false;

      llvm::DenseMap<mlir::Value, mlir::Value> map;
      if (!areSubgraphsEquivalent(slice0Val, sliceNVal, i, stride, map)) {
        return false;
      }
    }
    return true;
  }

  /// Recursively compares two subgraphs to determine if they are isomorphic
  /// with respect to a constant bit-stride.
  ///
  /// It assumes that all ExtractOp low-bit indices in the second subgraph
  /// are exactly (sliceIndex * stride) greater than those in the first.
  /// Caches results in slice0ToNMap to handle DAGs efficiently.
  bool areSubgraphsEquivalent(Value slice0Val, Value sliceNVal,
                              unsigned sliceIndex, int stride,
                              DenseMap<Value, Value> &slice0ToNMap) {

    if (slice0ToNMap.count(slice0Val))
      return slice0ToNMap[slice0Val] == sliceNVal;

    Operation *op0 = slice0Val.getDefiningOp();
    Operation *opN = sliceNVal.getDefiningOp();

    if (auto extract0 = dyn_cast_or_null<comb::ExtractOp>(op0)) {
      auto extractN = dyn_cast_or_null<comb::ExtractOp>(opN);

      if (extractN && extract0.getInput() == extractN.getInput() &&
          extractN.getLowBit() ==
              static_cast<unsigned>(static_cast<int>(extract0.getLowBit()) +
                                    static_cast<int>(sliceIndex) * stride)) {
        slice0ToNMap[slice0Val] = sliceNVal;
        return true;
      }
      return false;
    }

    if (slice0Val == sliceNVal && (mlir::isa<BlockArgument>(slice0Val) ||
                                   mlir::isa<hw::ConstantOp>(op0))) {
      slice0ToNMap[slice0Val] = sliceNVal;
      return true;
    }

    if (!op0 || !opN || op0->getName() != opN->getName() ||
        op0->getNumOperands() != opN->getNumOperands())
      return false;

    for (unsigned i = 0; i < op0->getNumOperands(); ++i) {
      if (!areSubgraphsEquivalent(op0->getOperand(i), opN->getOperand(i),
                                  sliceIndex, stride, slice0ToNMap))
        return false;
    }

    slice0ToNMap[slice0Val] = sliceNVal;
    return true;
  }

  /// Traverses through ConcatOps and basic logic gates to locate the
  /// original 1-bit source for a specific bit index.
  ///
  /// Returns the 1-bit Value or nullptr if the bit cannot be traced back
  /// to a concrete scalar source
  Value findBitSource(Value vectorVal, unsigned bitIndex) {

    if (auto blockArg = dyn_cast<BlockArgument>(vectorVal)) {
      if (blockArg.getType().isInteger(1))
        return blockArg;
      return nullptr;
    }

    Operation *op = vectorVal.getDefiningOp();

    if (op->getNumResults() == 1 && op->getResult(0).getType().isInteger(1)) {
      return op->getResult(0);
    }

    if (auto concat = dyn_cast<comb::ConcatOp>(op)) {
      unsigned currentBit = cast<IntegerType>(vectorVal.getType()).getWidth();
      for (Value operand : concat.getInputs()) {
        unsigned operandWidth = cast<IntegerType>(operand.getType()).getWidth();
        currentBit -= operandWidth;
        if (bitIndex >= currentBit && bitIndex < currentBit + operandWidth) {
          return findBitSource(operand, bitIndex - currentBit);
        }
      }
    } else if (auto orOp = dyn_cast<comb::OrOp>(op)) {
      if (orOp.getNumOperands() != 2)
        return nullptr;

      Value lhs = orOp.getInputs()[0];
      Value rhs = orOp.getInputs()[1];

      if (auto constRhs =
              dyn_cast_or_null<hw::ConstantOp>(rhs.getDefiningOp())) {
        if (!constRhs.getValue()[bitIndex])
          return findBitSource(lhs, bitIndex);
      }

      if (auto constLhs =
              dyn_cast_or_null<hw::ConstantOp>(lhs.getDefiningOp())) {
        if (!constLhs.getValue()[bitIndex])
          return findBitSource(rhs, bitIndex);
      }
    } else if (auto andOp = dyn_cast<comb::AndOp>(op)) {
      if (andOp.getNumOperands() != 2)
        return nullptr;

      Value lhs = andOp.getInputs()[0];
      Value rhs = andOp.getInputs()[1];

      if (auto constRhs =
              dyn_cast_or_null<hw::ConstantOp>(rhs.getDefiningOp())) {
        if (constRhs.getValue()[bitIndex])
          return findBitSource(lhs, bitIndex);
      }

      if (auto constLhs =
              dyn_cast_or_null<hw::ConstantOp>(lhs.getDefiningOp())) {
        if (constLhs.getValue()[bitIndex])
          return findBitSource(rhs, bitIndex);
      }
    }

    return nullptr;
  }

  /// Recursively builds the vectorized counterpart of a scalar subgraph.
  ///
  /// `scalarRoot` is the 1-bit root of the scalar bit-0 cone.
  /// `width` is the target vector width (= number of bit lanes).
  /// `map` caches already-vectorized scalar values to avoid duplicate work.
  ///
  /// Returns nullptr if any node in the subgraph cannot be vectorized.
  Value vectorizeSubgraph(OpBuilder &b, Value scalarRoot, unsigned width,
                          DenseMap<Value, Value> &map) {
    if (map.count(scalarRoot))
      return map[scalarRoot];

    // Base case: an ExtractOp represents one bit lane of a wider source vector.
    // Return that source vector directly; the other lanes are handled by the
    // isomorphic slices discovered in canVectorizeStructurally
    if (auto ex =
            dyn_cast_or_null<comb::ExtractOp>(scalarRoot.getDefiningOp())) {
      Value vec = ex.getInput();
      map[scalarRoot] = vec;
      return vec;
    }

    // Base case: a 1-bit constant or block argument (e.g., a shared selector)
    // must be broadcast to all `width` lanes via comb.replicate.
    if (isSafeSharedValue(scalarRoot)) {
      if (cast<IntegerType>(scalarRoot.getType()).getWidth() == 1)
        return comb::ReplicateOp::create(b, scalarRoot.getLoc(),
                                         b.getIntegerType(width), scalarRoot);
      // Wider constants are already the right width; pass through unchanged.
      return scalarRoot;
    }

    Operation *op = scalarRoot.getDefiningOp();
    if (!op)
      return nullptr;

    // Recursively vectorize all operands before creating the wide op.
    SmallVector<Value> ops;
    for (Value operand : op->getOperands()) {
      Value v = vectorizeSubgraph(b, operand, width, map);
      if (!v)
        return map[scalarRoot] = nullptr;
      ops.push_back(v);
    }

    Type vecTy = b.getIntegerType(width);
    Value result;

    // Lift the scalar op to its N-bit equivalent.
    if (isa<comb::AndOp>(op))
      result = comb::AndOp::create(b, op->getLoc(), vecTy, ops);
    else if (isa<comb::OrOp>(op))
      result = comb::OrOp::create(b, op->getLoc(), vecTy, ops);
    else if (isa<comb::XorOp>(op))
      result = comb::XorOp::create(b, op->getLoc(), vecTy, ops);
    else if (auto muxOp = dyn_cast<comb::MuxOp>(op)) {
      Value sel = muxOp.getCond();
      result = comb::MuxOp::create(b, muxOp.getLoc(), sel, ops[1], ops[2]);
    } else
      // Unsupported op kind; signal failure to the caller.
      return nullptr;

    map[scalarRoot] = result;
    return result;
  }

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
            bitArrays[extractOp.getResult()] = bits;
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
            bitArrays[concatOp.getResult()] = concatenatedArray;
          })
          .Case<comb::AndOp, comb::OrOp, comb::XorOp, comb::MuxOp>(
              [&](Operation *op) {
                auto result = op->getResult(0);
                auto resultType = dyn_cast<IntegerType>(result.getType());
                if (resultType && resultType.getWidth() == 1) {
                  BitArray arr;
                  arr.bits.push_back(Bit(result, 0));
                  bitArrays[result] = arr;
                }
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
