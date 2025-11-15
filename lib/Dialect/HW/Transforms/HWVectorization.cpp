//===- HWVectorization.cpp - HW Vectorization Pass ----------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass performs structural vectorization of hardware modules,
// merging scalar bit-level assignments into vectorized operations.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <algorithm>

namespace circt {
namespace hw {
#define GEN_PASS_DEF_HWVECTORIZATION
#include "circt/Dialect/HW/Passes.h.inc"
} // namespace hw
} // namespace circt

const int INDEX_SENTINEL_VALUE = -1;
const int TOMBSTONE_SENTINEL_VALUE = -1;

struct bit {
  mlir::Value source;
  int index;

  bit(mlir::Value source, int index);
  bit();
  bit(const bit &other);

  bit &operator=(const bit &other);
  bool operator==(const bit &other) const;

  bool left_adjacent(const bit &other);
  bool right_adjacent(const bit &other);
};

bit::bit(mlir::Value source, int index) : source(source), index(index) {}

bit::bit() : source(mlir::Value()), index(0) {}

bit::bit(const bit &other) : source(other.source), index(other.index) {}

bit &bit::operator=(const bit &other) {
  if (this == &other)
    return *this;
  source = other.source;
  index = other.index;
  return *this;
}

bool bit::operator==(const bit &other) const {
  return source == other.source and index == other.index;
}

bool bit::left_adjacent(const bit &other) {
  return source == other.source and index == other.index + 1;
}

bool bit::right_adjacent(const bit &other) {
  return source == other.source and index == other.index - 1;
}

namespace llvm {
inline hash_code bit_hash_code(const bit &b) {
  return llvm::hash_combine(b.source, b.index);
}
template <>
struct DenseMapInfo<bit> {
  static inline bit getEmptyKey() {
    return bit(llvm::DenseMapInfo<mlir::Value>::getEmptyKey(),
               INDEX_SENTINEL_VALUE);
  }
  static inline bit getTombstoneKey() {
    return bit(llvm::DenseMapInfo<mlir::Value>::getEmptyKey(),
               TOMBSTONE_SENTINEL_VALUE);
  }
  static unsigned getHashValue(const bit &A) {
    return static_cast<unsigned>(bit_hash_code(A));
  }
  static bool isEqual(const bit &A, const bit &B) { return A == B; }
};
} // namespace llvm

namespace {

using namespace mlir;
using namespace circt;
using namespace comb;
using namespace hw;

struct bit_array {
  llvm::DenseMap<int, bit> bits;

  bit_array(llvm::DenseMap<int, bit> &bits);
  bit_array(const bit_array &other);
  bit_array();

  bit get_bit(int index);

  bool all_bits_have_same_source() const;
  bool is_linear(int size, mlir::Value sourceInput);
  bool is_reverse(int size, mlir::Value sourceInput);

  mlir::Value getSingleSourceValue() const;
  size_t size() const;
};

class vectorizer {
public:
  vectorizer(hw::HWModuleOp module);

  hw::HWModuleOp module;
  llvm::DenseMap<mlir::Value, bit_array> bit_arrays;

  mlir::Value findBitSource(mlir::Value vectorVal, unsigned bitIndex,
                            int depth = 0);
  mlir::Value
  vectorizeSubgraph(OpBuilder &builder, mlir::Value slice0Val,
                    unsigned vectorWidth,
                    llvm::DenseMap<mlir::Value, mlir::Value> &vectorizedMap);

  bool can_vectorize_structurally(mlir::Value output);
  bool areSubgraphsEquivalent(
      mlir::Value slice0Val, mlir::Value sliceNVal, unsigned sliceIndex,
      int stride, llvm::DenseMap<mlir::Value, mlir::Value> &slice0ToNMap);
  bool isValidPermutation(const llvm::SmallVector<unsigned> &perm,
                          unsigned bitWidth);
  bool can_apply_partial_vectorization(Value oldOutputVal);

  bool hasCrossBitDependencies(mlir::Value outputVal);
  void collectLogicCone(mlir::Value val, llvm::DenseSet<mlir::Value> &cone);
  bool isSafeSharedValue(mlir::Value val,
                         llvm::SmallPtrSetImpl<mlir::Value> &visited);
  bool isSafeSharedValue(mlir::Value val);

  void process_extract_ops();
  void process_concat_ops();

  void process_or_op(comb::OrOp op);
  void process_and_op(comb::AndOp op);
  void process_logical_ops();
  void process_xor_op(comb::XorOp op);

  void vectorize();

  void apply_linear_vectorization(mlir::Value oldOutputVal,
                                  mlir::Value sourceInput);
  void apply_reverse_vectorization(mlir::OpBuilder &builder,
                                   mlir::Value oldOutputVal,
                                   mlir::Value sourceInput);
  void apply_mix_vectorization(mlir::OpBuilder &builder,
                               mlir::Value oldOutputVal,
                               mlir::Value sourceInput,
                               const llvm::SmallVector<unsigned> &map);
  void apply_structural_vectorization(OpBuilder &builder,
                                      mlir::Value oldOutputVal);
  void apply_partial_vectorization(OpBuilder &builder,
                                   mlir::Value oldOutputVal);

  bool cleanup_dead_ops(Block &body);
};

bit_array::bit_array(llvm::DenseMap<int, bit> &bits) : bits(bits) {}
bit_array::bit_array(const bit_array &other) : bits(other.bits) {}
bit_array::bit_array() : bits(llvm::DenseMap<int, bit>()) {}

bool bit_array::all_bits_have_same_source() const {
  llvm::DenseSet<mlir::Value> sources;
  for (const auto &[_, bit] : bits) {
    if (!sources.contains(bit.source))
      sources.insert(bit.source);
    if (sources.size() >= 2)
      return false;
  }
  return true;
}

bool bit_array::is_linear(int size, mlir::Value sourceInput) {
  if (bits.size() != (unsigned)size)
    return false;
  for (const auto &[index, bit] : bits) {
    if (bit.source != sourceInput || bit.index != index) {
      return false;
    }
  }
  return true;
}

bool bit_array::is_reverse(int size, mlir::Value sourceInput) {
  if (bits.size() != (unsigned)size)
    return false;
  for (const auto &[index, bit] : bits) {
    if (bit.source != sourceInput || (size - 1) - index != bit.index) {
      return false;
    }
  }
  return true;
}

bit bit_array::get_bit(int n) { return bits[n]; }

mlir::Value bit_array::getSingleSourceValue() const {
  if (!all_bits_have_same_source() || bits.empty()) {
    return nullptr;
  }
  return bits.begin()->second.source;
}

size_t bit_array::size() const { return bits.size(); }

vectorizer::vectorizer(hw::HWModuleOp module) : module(module) {}

void vectorizer::vectorize() {
  process_extract_ops();
  process_concat_ops();
  process_logical_ops();

  Block &block = module.getBody().front();
  auto outputOp = dyn_cast<hw::OutputOp>(block.getTerminator());
  if (!outputOp)
    return;

  OpBuilder builder(module.getContext());
  bool changed = false;

  for (Value oldOutputVal : outputOp->getOperands()) {
    bool transformed = false;
    unsigned bitWidth = cast<IntegerType>(oldOutputVal.getType()).getWidth();

    if (bit_arrays.count(oldOutputVal)) {
      bit_array &arr = bit_arrays[oldOutputVal];
      if (arr.size() == bitWidth) {
        Value sourceInput = arr.getSingleSourceValue();
        if (sourceInput) {
          llvm::SmallVector<unsigned> currentPermutationMap;
          for (unsigned i = 0; i < bitWidth; ++i) {
            currentPermutationMap.push_back(arr.get_bit(i).index);
          }

          if (isValidPermutation(currentPermutationMap, bitWidth)) {
            if (arr.is_linear(bitWidth, sourceInput)) {
              apply_linear_vectorization(oldOutputVal, sourceInput);
              transformed = true;
            } else if (arr.is_reverse(bitWidth, sourceInput)) {
              apply_reverse_vectorization(builder, oldOutputVal, sourceInput);
              transformed = true;
            } else {
              apply_mix_vectorization(builder, oldOutputVal, sourceInput,
                                      currentPermutationMap);
              transformed = true;
            }
          }
        }
      }
    }

    if (!transformed) {
      if (hasCrossBitDependencies(oldOutputVal)) {
        continue;
      } else if (can_vectorize_structurally(oldOutputVal)) {
        Value bit0Source = findBitSource(oldOutputVal, 0);
        Value bit1Source = findBitSource(oldOutputVal, 1);

        auto extract0 =
            bit0Source ? bit0Source.getDefiningOp<comb::ExtractOp>() : nullptr;
        auto extract1 =
            bit1Source ? bit1Source.getDefiningOp<comb::ExtractOp>() : nullptr;

        bool patternApplied = false;
        if (extract0 && extract1 &&
            extract0.getInput() == extract1.getInput()) {
          Value sourceInput = extract0.getInput();
          int lowBit0 = extract0.getLowBit();
          int lowBit1 = extract1.getLowBit();

          if (lowBit1 == lowBit0 + 1) {
            apply_linear_vectorization(oldOutputVal, sourceInput);
            patternApplied = true;
          } else if (lowBit1 == lowBit0 - 1) {
            apply_reverse_vectorization(builder, oldOutputVal, sourceInput);
            patternApplied = true;
          }
        }

        if (!patternApplied) {
          apply_structural_vectorization(builder, oldOutputVal);
        }
        transformed = true;
      }
    }

    if (!transformed && can_apply_partial_vectorization(oldOutputVal)) {
      apply_partial_vectorization(builder, oldOutputVal);
      transformed = true;
    }

    if (transformed)
      changed = true;
  }

  if (changed) {
    cleanup_dead_ops(block);
  }
}

void vectorizer::process_logical_ops() {
  module.walk([&](mlir::Operation *op) {
    if (llvm::isa<comb::OrOp, comb::AndOp, comb::XorOp>(op)) {
      if (auto or_op = llvm::dyn_cast<comb::OrOp>(op)) {
        process_or_op(or_op);
      } else if (auto and_op = llvm::dyn_cast<comb::AndOp>(op)) {
        process_and_op(and_op);
      } else {
        auto xor_op = llvm::dyn_cast<comb::XorOp>(op);
        process_xor_op(xor_op);
      }
    }
  });
}

void vectorizer::process_xor_op(comb::XorOp op) {
  mlir::Value result = op.getResult();
  bit_arrays.insert({result, bit_array()});
}

void vectorizer::process_or_op(comb::OrOp op) {
  mlir::Value result = op.getResult();
  bit_arrays.insert({result, bit_array()});
}

void vectorizer::process_and_op(comb::AndOp op) {
  mlir::Value result = op.getResult();
  bit_arrays.insert({result, bit_array()});
}

void vectorizer::process_extract_ops() {
  module.walk([&](comb::ExtractOp op) {
    mlir::Value input = op.getInput();
    mlir::Value result = op.getResult();
    int index = op.getLowBit();
    llvm::DenseMap<int, bit> bit_dense_map;
    bit_dense_map.insert({0, bit(input, index)});
    bit_array bits(bit_dense_map);
    bit_arrays.insert({result, bits});
  });
}

void vectorizer::process_concat_ops() {
  module.walk([&](comb::ConcatOp op) {
    mlir::Value result = op.getResult();
    bit_array concatenatedArray;
    unsigned currentBitOffset = 0;

    for (Value operand : llvm::reverse(op.getInputs())) {
      unsigned operandWidth = cast<IntegerType>(operand.getType()).getWidth();

      if (bit_arrays.count(operand)) {
        bit_array &operandArray = bit_arrays[operand];
        for (auto const &[bitIndex, bitInfo] : operandArray.bits) {
          concatenatedArray.bits[bitIndex + currentBitOffset] = bitInfo;
        }
      }
      currentBitOffset += operandWidth;
    }
    bit_arrays.insert({result, concatenatedArray});
  });
}

void vectorizer::apply_linear_vectorization(Value oldOutputVal,
                                            Value sourceInput) {
  oldOutputVal.replaceAllUsesWith(sourceInput);
}

void vectorizer::apply_reverse_vectorization(OpBuilder &builder,
                                             Value oldOutputVal,
                                             Value sourceInput) {
  builder.setInsertionPoint(*oldOutputVal.getUsers().begin());
  Location loc = sourceInput.getLoc();
  Value reversedInput =
      comb::ReverseOp::create(builder, loc, sourceInput.getType(), sourceInput);
  oldOutputVal.replaceAllUsesWith(reversedInput);
}

void vectorizer::apply_mix_vectorization(
    OpBuilder &builder, Value oldOutputVal, Value sourceInput,
    const llvm::SmallVector<unsigned> &map) {
  unsigned bitWidth = cast<IntegerType>(oldOutputVal.getType()).getWidth();
  Location loc = sourceInput.getLoc();
  builder.setInsertionPoint(*oldOutputVal.getUsers().begin());

  llvm::SmallVector<Value> extractedChunks;
  unsigned i = 0;
  while (i < bitWidth) {
    unsigned startBit = map[i];
    unsigned len = 1;
    while ((i + len < bitWidth) && (map[i + len] == startBit + len)) {
      len++;
    }
    Value chunk = comb::ExtractOp::create(
        builder, loc, builder.getIntegerType(len), sourceInput,
        builder.getI32IntegerAttr(startBit));
    extractedChunks.push_back(chunk);
    i += len;
  }

  Value newOutputVal;
  if (extractedChunks.size() == 1) {
    newOutputVal = extractedChunks[0];
  } else {
    std::reverse(extractedChunks.begin(), extractedChunks.end());
    unsigned totalWidth = 0;
    for (Value chunk : extractedChunks)
      totalWidth += cast<IntegerType>(chunk.getType()).getWidth();
    Type resultType = builder.getIntegerType(totalWidth);

    newOutputVal =
        comb::ConcatOp::create(builder, loc, resultType, extractedChunks);
  }

  oldOutputVal.replaceAllUsesWith(newOutputVal);
}

void vectorizer::apply_structural_vectorization(OpBuilder &builder,
                                                mlir::Value oldOutputVal) {
  unsigned bitWidth = cast<IntegerType>(oldOutputVal.getType()).getWidth();
  Value slice0Val = findBitSource(oldOutputVal, 0);
  if (!slice0Val)
    return;

  llvm::DenseMap<mlir::Value, mlir::Value> vectorizedMap;
  builder.setInsertionPoint(*oldOutputVal.getUsers().begin());

  Value newOutputVal =
      vectorizeSubgraph(builder, slice0Val, bitWidth, vectorizedMap);
  if (!newOutputVal)
    return;

  oldOutputVal.replaceAllUsesWith(newOutputVal);
}

void vectorizer::apply_partial_vectorization(OpBuilder &builder,
                                             mlir::Value oldOutputVal) {
  unsigned bitWidth = cast<IntegerType>(oldOutputVal.getType()).getWidth();
  Location loc = oldOutputVal.getLoc();

  if (oldOutputVal.use_empty())
    return;

  builder.setInsertionPoint(*oldOutputVal.getUsers().begin());

  SmallVector<Value> chunks;
  for (int i = bitWidth - 1; i >= 0;) {
    Value bitSource = findBitSource(oldOutputVal, i);
    if (!bitSource)
      return;

    Operation *sourceOp = bitSource.getDefiningOp();
    int len = 1;

    if (auto extractOp = dyn_cast_or_null<comb::ExtractOp>(sourceOp)) {
      while ((i - len) >= 0) {
        Value nextBitSource = findBitSource(oldOutputVal, i - len);
        auto nextExtractOp =
            dyn_cast_or_null<comb::ExtractOp>(nextBitSource.getDefiningOp());

        if (nextExtractOp && nextExtractOp.getInput() == extractOp.getInput() &&
            nextExtractOp.getLowBit() == extractOp.getLowBit() - len) {
          len++;
        } else {
          break;
        }
      }
      Value sourceVec = extractOp.getInput();
      unsigned extractLowBit = extractOp.getLowBit() - (len - 1);
      Value extractedChunk = comb::ExtractOp::create(
          builder, loc, builder.getIntegerType(len), sourceVec,
          builder.getI32IntegerAttr(extractLowBit));
      chunks.push_back(extractedChunk);
    } else {
      chunks.push_back(bitSource);
    }
    i -= len;
  }

  if (chunks.size() == 1 &&
      cast<IntegerType>(chunks[0].getType()).getWidth() == bitWidth) {
    oldOutputVal.replaceAllUsesWith(chunks[0]);
    return;
  }

  unsigned totalWidth = 0;
  for (Value chunk : chunks)
    totalWidth += cast<IntegerType>(chunk.getType()).getWidth();
  Type resultType = builder.getIntegerType(totalWidth);

  Value newOutputVal = comb::ConcatOp::create(builder, loc, resultType, chunks);

  oldOutputVal.replaceAllUsesWith(newOutputVal);
}

bool vectorizer::hasCrossBitDependencies(mlir::Value outputVal) {
  unsigned bitWidth = cast<IntegerType>(outputVal.getType()).getWidth();
  if (bitWidth <= 1)
    return false;

  llvm::SmallVector<llvm::DenseSet<mlir::Value>> bitCones(bitWidth);
  for (unsigned i = 0; i < bitWidth; ++i) {
    mlir::Value bitSource = findBitSource(outputVal, i);
    if (bitSource) {
      collectLogicCone(bitSource, bitCones[i]);
    }
  }

  for (unsigned i = 0; i < bitWidth; ++i) {
    for (unsigned j = i + 1; j < bitWidth; ++j) {
      for (mlir::Value val : bitCones[i]) {
        if (bitCones[j].count(val)) {
          if (!isSafeSharedValue(val)) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

bool vectorizer::can_vectorize_structurally(mlir::Value output) {
  unsigned bitWidth = cast<IntegerType>(output.getType()).getWidth();
  if (bitWidth <= 1)
    return false;

  Value slice0Val = findBitSource(output, 0);
  if (!slice0Val || !slice0Val.getDefiningOp())
    return false;

  Value slice1Val = findBitSource(output, 1);
  if (!slice1Val || !slice1Val.getDefiningOp())
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

bool vectorizer::can_apply_partial_vectorization(Value oldOutputVal) {
  unsigned bitWidth = cast<IntegerType>(oldOutputVal.getType()).getWidth();
  if (bitWidth <= 1)
    return false;

  for (unsigned i = 0; i < bitWidth; ++i) {
    if (!findBitSource(oldOutputVal, i)) {
      return false;
    }
  }
  return true;
}

bool vectorizer::isValidPermutation(const llvm::SmallVector<unsigned> &perm,
                                    unsigned bitWidth) {
  if (perm.size() != bitWidth)
    return false;
  llvm::SmallVector<bool> seen(bitWidth, false);

  for (unsigned idx : perm) {
    if (idx >= bitWidth)
      return false;
    if (seen[idx])
      return false;
    seen[idx] = true;
  }
  return true;
}

bool vectorizer::isSafeSharedValue(mlir::Value val) {
  llvm::SmallPtrSet<mlir::Value, 8> visited;
  return isSafeSharedValue(val, visited);
}

void vectorizer::collectLogicCone(mlir::Value val,
                                  llvm::DenseSet<mlir::Value> &cone) {
  if (cone.count(val)) {
    return;
  }
  cone.insert(val);

  Operation *definingOp = val.getDefiningOp();
  if (!definingOp || isa<BlockArgument>(val) ||
      isa<hw::ConstantOp>(definingOp)) {
    return;
  }

  for (Value operand : definingOp->getOperands()) {
    collectLogicCone(operand, cone);
  }
}

bool vectorizer::isSafeSharedValue(
    mlir::Value val, llvm::SmallPtrSetImpl<mlir::Value> &visited) {
  if (!val || isa<BlockArgument>(val) || val.getDefiningOp<hw::ConstantOp>())
    return true;

  if (!visited.insert(val).second)
    return true;

  if (auto *op = val.getDefiningOp()) {
    for (auto operand : op->getOperands()) {
      if (!isSafeSharedValue(operand, visited))
        return false;
    }
    return true;
  }
  return false;
}

bool vectorizer::areSubgraphsEquivalent(
    mlir::Value slice0Val, mlir::Value sliceNVal, unsigned sliceIndex,
    int stride, llvm::DenseMap<mlir::Value, mlir::Value> &slice0ToNMap) {
  if (slice0ToNMap.count(slice0Val))
    return slice0ToNMap[slice0Val] == sliceNVal;

  Operation *op0 = slice0Val.getDefiningOp();
  Operation *opN = sliceNVal.getDefiningOp();

  if (auto extract0 = dyn_cast_or_null<comb::ExtractOp>(op0)) {
    auto extractN = dyn_cast_or_null<comb::ExtractOp>(opN);

    if (extractN && extract0.getInput() == extractN.getInput() &&
        extractN.getLowBit() ==
            (unsigned)((int)extract0.getLowBit() + (int)sliceIndex * stride)) {
      slice0ToNMap[slice0Val] = sliceNVal;
      return true;
    }
    return false;
  }

  if (slice0Val == sliceNVal &&
      (mlir::isa<BlockArgument>(slice0Val) || mlir::isa<hw::ConstantOp>(op0))) {
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

mlir::Value vectorizer::vectorizeSubgraph(
    OpBuilder &builder, mlir::Value slice0Val, unsigned vectorWidth,
    llvm::DenseMap<mlir::Value, mlir::Value> &vectorizedMap) {
  if (vectorizedMap.count(slice0Val))
    return vectorizedMap[slice0Val];

  if (auto extractOp =
          dyn_cast_or_null<comb::ExtractOp>(slice0Val.getDefiningOp())) {
    Value vector = extractOp.getInput();
    vectorizedMap[slice0Val] = vector;
    return vector;
  }

  if (mlir::isa<BlockArgument>(slice0Val) ||
      mlir::isa<hw::ConstantOp>(slice0Val.getDefiningOp())) {
    unsigned scalarWidth = cast<IntegerType>(slice0Val.getType()).getWidth();
    if (scalarWidth == 1) {
      return comb::ReplicateOp::create(builder, slice0Val.getLoc(),
                                       builder.getIntegerType(vectorWidth),
                                       slice0Val);
    }
    return slice0Val;
  }

  Operation *op0 = slice0Val.getDefiningOp();
  if (!op0)
    return nullptr;
  Location loc = op0->getLoc();

  SmallVector<Value> vectorizedOperands;
  for (Value operand : op0->getOperands()) {
    Value vectorizedOperand =
        vectorizeSubgraph(builder, operand, vectorWidth, vectorizedMap);
    if (!vectorizedOperand)
      return nullptr;
    vectorizedOperands.push_back(vectorizedOperand);
  }

  Type resultType = builder.getIntegerType(vectorWidth);
  Value vectorizedResult;

  if (dyn_cast<comb::AndOp>(op0)) {
    vectorizedResult =
        comb::AndOp::create(builder, loc, resultType, vectorizedOperands);
  } else if (dyn_cast<comb::OrOp>(op0)) {
    vectorizedResult =
        comb::OrOp::create(builder, loc, resultType, vectorizedOperands);
  } else if (dyn_cast<comb::XorOp>(op0)) {
    vectorizedResult =
        comb::XorOp::create(builder, loc, resultType, vectorizedOperands);
  } else if (dyn_cast<comb::MuxOp>(op0)) {
    Value sel = vectorizedOperands[0];
    if (cast<IntegerType>(sel.getType()).getWidth() != 1) {
      sel = comb::ExtractOp::create(builder, loc, builder.getI1Type(), sel, 0);
    }
    Value replicatedSel =
        comb::ReplicateOp::create(builder, loc, resultType, sel);
    vectorizedResult =
        comb::MuxOp::create(builder, loc, replicatedSel, vectorizedOperands[1],
                            vectorizedOperands[2]);
  } else {
    return nullptr;
  }

  vectorizedMap[slice0Val] = vectorizedResult;
  return vectorizedResult;
}

mlir::Value vectorizer::findBitSource(mlir::Value vectorVal, unsigned bitIndex,
                                      int depth) {
  if (auto blockArg = dyn_cast<BlockArgument>(vectorVal)) {
    if (blockArg.getType().isInteger(1)) {
      return blockArg;
    }
    return nullptr;
  }

  Operation *op = vectorVal.getDefiningOp();
  if (!op) {
    return nullptr;
  }

  if (op->getNumResults() == 1 && op->getResult(0).getType().isInteger(1)) {
    return op->getResult(0);
  }

  if (auto constOp = dyn_cast<hw::ConstantOp>(op)) {
    if (constOp.getType().isInteger(1)) {
      return constOp.getResult();
    }
    return nullptr;
  }

  if (auto concat = dyn_cast<comb::ConcatOp>(op)) {
    unsigned currentBit = cast<IntegerType>(vectorVal.getType()).getWidth();
    for (Value operand : concat.getInputs()) {
      unsigned operandWidth = cast<IntegerType>(operand.getType()).getWidth();
      currentBit -= operandWidth;
      if (bitIndex >= currentBit && bitIndex < currentBit + operandWidth) {
        return findBitSource(operand, bitIndex - currentBit, depth + 1);
      }
    }
  } else if (auto orOp = dyn_cast<comb::OrOp>(op)) {
    if (auto source = findBitSource(orOp.getInputs()[1], bitIndex, depth + 1)) {
      if (auto sourceConst =
              dyn_cast_or_null<hw::ConstantOp>(source.getDefiningOp())) {
        if (!sourceConst.getValue().isZero())
          return source;
      } else {
        return source;
      }
    }
    return findBitSource(orOp.getInputs()[0], bitIndex, depth + 1);
  } else if (auto andOp = dyn_cast<comb::AndOp>(op)) {
    Value lhs = andOp.getInputs()[0];
    Value rhs = andOp.getInputs()[1];
    if (isa_and_nonnull<hw::ConstantOp>(rhs.getDefiningOp()))
      return findBitSource(lhs, bitIndex, depth + 1);
    if (isa_and_nonnull<hw::ConstantOp>(lhs.getDefiningOp()))
      return findBitSource(rhs, bitIndex, depth + 1);
  }

  return nullptr;
}

bool vectorizer::cleanup_dead_ops(Block &block) {
  bool overallChanged = false;
  bool changedInIteration = true;
  while (changedInIteration) {
    changedInIteration = false;
    llvm::SmallVector<Operation *, 16> deadOps;
    for (Operation &op : block) {
      if (op.use_empty() && !op.hasTrait<mlir::OpTrait::IsTerminator>()) {
        deadOps.push_back(&op);
      }
    }
    if (!deadOps.empty()) {
      changedInIteration = true;
      overallChanged = true;
      for (Operation *op : deadOps) {
        op->erase();
      }
    }
  }
  return overallChanged;
}

struct HWVectorizationPass
    : public hw::impl::HWVectorizationBase<HWVectorizationPass> {

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<circt::comb::CombDialect, circt::hw::HWDialect,
                    circt::sv::SVDialect>();
  }

  void runOnOperation() override {
    hw::HWModuleOp module = getOperation();

    bool containsLLHD = false;
    module.walk([&](mlir::Operation *op) {
      if (op->getDialect()->getNamespace() == "llhd") {
        containsLLHD = true;
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });

    if (containsLLHD) {
      return;
    }

    vectorizer v(module);
    v.vectorize();
  }
};

} // namespace
