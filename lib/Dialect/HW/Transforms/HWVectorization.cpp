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
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"
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

struct Bit {
  mlir::Value source;
  int index;

  Bit(mlir::Value source, int index);
  Bit();
  Bit(const Bit &other);

  Bit &operator=(const Bit &other);
  bool operator==(const Bit &other) const;

  bool leftAdjacent(const Bit &other);
  bool rightAdjacent(const Bit &other);
};

Bit::Bit(mlir::Value source, int index) : source(source), index(index) {}

Bit::Bit() : source(mlir::Value()), index(0) {}

Bit::Bit(const Bit &other) : source(other.source), index(other.index) {}

Bit &Bit::operator=(const Bit &other) {
  if (this == &other)
    return *this;
  source = other.source;
  index = other.index;
  return *this;
}

bool Bit::operator==(const Bit &other) const {
  return source == other.source and index == other.index;
}

bool Bit::leftAdjacent(const Bit &other) {
  return source == other.source and index == other.index + 1;
}

bool Bit::rightAdjacent(const Bit &other) {
  return source == other.source and index == other.index - 1;
}

namespace llvm {
inline hash_code bitHashCode(const Bit &b) {
  return llvm::hash_combine(b.source, b.index);
}
template <>
struct DenseMapInfo<Bit> {
  static inline Bit getEmptyKey() {
    return Bit(llvm::DenseMapInfo<mlir::Value>::getEmptyKey(),
               INDEX_SENTINEL_VALUE);
  }
  static inline Bit getTombstoneKey() {
    return Bit(llvm::DenseMapInfo<mlir::Value>::getEmptyKey(),
               TOMBSTONE_SENTINEL_VALUE);
  }
  static unsigned getHashValue(const Bit &A) {
    return static_cast<unsigned>(bitHashCode(A));
  }
  static bool isEqual(const Bit &A, const Bit &B) { return A == B; }
};
} // namespace llvm

namespace {

using namespace mlir;
using namespace circt;
using namespace comb;
using namespace hw;

struct BitArray {
  llvm::DenseMap<int, Bit> bits;

  BitArray(llvm::DenseMap<int, Bit> &bits);
  BitArray(const BitArray &other);
  BitArray();

  Bit getBit(int index);

  bool allBitsHaveSameSource() const;
  bool isLinear(int size, mlir::Value sourceInput);
  bool isReverse(int size, mlir::Value sourceInput);

  mlir::Value getSingleSourceValue() const;
  size_t size() const;
};

class Vectorizer {
public:
  Vectorizer(hw::HWModuleOp module);

  hw::HWModuleOp module;
  llvm::DenseMap<mlir::Value, BitArray> bitArrays;

  bool isBitConstant(mlir::Value val, unsigned bitIndex, bool expectedVal);

  mlir::Value findBitSource(mlir::Value vectorVal, unsigned bitIndex,
                            int depth = 0);
  mlir::Value
  vectorizeSubgraph(OpBuilder &builder, mlir::Value slice0Val,
                    unsigned vectorWidth,
                    llvm::DenseMap<mlir::Value, mlir::Value> &vectorizedMap);

  bool canVectorizeStructurally(mlir::Value output);
  bool areSubgraphsEquivalent(
      mlir::Value slice0Val, mlir::Value sliceNVal, unsigned sliceIndex,
      int stride, llvm::DenseMap<mlir::Value, mlir::Value> &slice0ToNMap);
  bool isValidPermutation(const llvm::SmallVector<unsigned> &perm,
                          unsigned bitWidth);
  bool canApplyPartialVectorization(Value oldOutputVal);

  bool hasCrossBitDependencies(mlir::Value outputVal);
  void collectLogicCone(mlir::Value val, llvm::DenseSet<mlir::Value> &cone);
  bool isSafeSharedValue(mlir::Value val,
                         llvm::SmallPtrSetImpl<mlir::Value> &visited);
  bool isSafeSharedValue(mlir::Value val);

  void processExtractOps();
  void processConcatOps();

  void processOrOp(comb::OrOp op);
  void processAndOp(comb::AndOp op);
  void processLogicalOps();
  void processXorOp(comb::XorOp op);

  void vectorize();

  void applyLinearVectorization(mlir::Value oldOutputVal,
                                mlir::Value sourceInput);
  void applyReverseVectorization(mlir::OpBuilder &builder,
                                 mlir::Value oldOutputVal,
                                 mlir::Value sourceInput);
  void applyMixVectorization(mlir::OpBuilder &builder, mlir::Value oldOutputVal,
                             mlir::Value sourceInput,
                             const llvm::SmallVector<unsigned> &map);
  void applyStructuralVectorization(OpBuilder &builder,
                                    mlir::Value oldOutputVal);
  void applyPartialVectorization(OpBuilder &builder, mlir::Value oldOutputVal);
};

BitArray::BitArray(llvm::DenseMap<int, Bit> &bits) : bits(bits) {}
BitArray::BitArray(const BitArray &other) : bits(other.bits) {}
BitArray::BitArray() : bits(llvm::DenseMap<int, Bit>()) {}

bool BitArray::allBitsHaveSameSource() const {
  mlir::Value source;
  for (const auto &[_, bit] : bits) {
    if (source && source != bit.source)
      return false;
    source = bit.source;
  }
  return true;
}

bool BitArray::isLinear(int size, mlir::Value sourceInput) {
  if (bits.size() != (unsigned)size)
    return false;
  for (const auto &[index, bit] : bits) {
    if (bit.source != sourceInput || bit.index != index) {
      return false;
    }
  }
  return true;
}

bool BitArray::isReverse(int size, mlir::Value sourceInput) {
  if (bits.size() != (unsigned)size)
    return false;
  for (const auto &[index, bit] : bits) {
    if (bit.source != sourceInput || (size - 1) - index != bit.index) {
      return false;
    }
  }
  return true;
}

Bit BitArray::getBit(int n) { return bits[n]; }

mlir::Value BitArray::getSingleSourceValue() const {
  if (!allBitsHaveSameSource() || bits.empty()) {
    return nullptr;
  }
  return bits.begin()->second.source;
}

size_t BitArray::size() const { return bits.size(); }

Vectorizer::Vectorizer(hw::HWModuleOp module) : module(module) {}

void Vectorizer::vectorize() {
  processExtractOps();
  processConcatOps();
  processLogicalOps();

  Block &block = module.getBody().front();
  auto outputOp = dyn_cast<hw::OutputOp>(block.getTerminator());

  IRRewriter rewriter(module.getContext());
  bool changed = false;

  for (Value oldOutputVal : outputOp->getOperands()) {
    bool transformed = false;
    unsigned bitWidth = cast<IntegerType>(oldOutputVal.getType()).getWidth();

    if (bitArrays.count(oldOutputVal)) {
      BitArray &arr = bitArrays[oldOutputVal];
      if (arr.size() == bitWidth) {
        Value sourceInput = arr.getSingleSourceValue();
        if (sourceInput) {
          llvm::SmallVector<unsigned> currentPermutationMap;
          for (unsigned i = 0; i < bitWidth; ++i) {
            currentPermutationMap.push_back(arr.getBit(i).index);
          }

          if (isValidPermutation(currentPermutationMap, bitWidth)) {
            if (arr.isLinear(bitWidth, sourceInput)) {
              applyLinearVectorization(oldOutputVal, sourceInput);
              transformed = true;
            } else if (arr.isReverse(bitWidth, sourceInput)) {
              applyReverseVectorization(rewriter, oldOutputVal, sourceInput);
              transformed = true;
            } else {
              applyMixVectorization(rewriter, oldOutputVal, sourceInput,
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
      } else if (canVectorizeStructurally(oldOutputVal)) {
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
            applyLinearVectorization(oldOutputVal, sourceInput);
            patternApplied = true;
          } else if (lowBit1 == lowBit0 - 1) {
            applyReverseVectorization(rewriter, oldOutputVal, sourceInput);
            patternApplied = true;
          }
        }

        if (!patternApplied) {
          applyStructuralVectorization(rewriter, oldOutputVal);
        }
        transformed = true;
      }
    }

    if (!transformed && canApplyPartialVectorization(oldOutputVal)) {
      applyPartialVectorization(rewriter, oldOutputVal);
      transformed = true;
    }

    if (transformed)
      changed = true;
  }

  if (changed) {
    (void)mlir::runRegionDCE(rewriter, module.getBody());
  }
}

void Vectorizer::processLogicalOps() {
  module.walk([&](mlir::Operation *op) {
    if (llvm::isa<comb::OrOp, comb::AndOp, comb::XorOp>(op)) {
      if (auto orOp = llvm::dyn_cast<comb::OrOp>(op)) {
        processOrOp(orOp);
      } else if (auto andOp = llvm::dyn_cast<comb::AndOp>(op)) {
        processAndOp(andOp);
      } else {
        auto xorOp = llvm::dyn_cast<comb::XorOp>(op);
        processXorOp(xorOp);
      }
    }
  });
}

void Vectorizer::processXorOp(comb::XorOp op) {
  mlir::Value result = op.getResult();
  bitArrays.insert({result, BitArray()});
}

void Vectorizer::processOrOp(comb::OrOp op) {
  mlir::Value result = op.getResult();
  bitArrays.insert({result, BitArray()});
}

void Vectorizer::processAndOp(comb::AndOp op) {
  mlir::Value result = op.getResult();
  bitArrays.insert({result, BitArray()});
}

void Vectorizer::processExtractOps() {
  module.walk([&](comb::ExtractOp op) {
    mlir::Value input = op.getInput();
    mlir::Value result = op.getResult();
    int index = op.getLowBit();
    llvm::DenseMap<int, Bit> bitDenseMap;
    bitDenseMap.insert({0, Bit(input, index)});
    BitArray bits(bitDenseMap);
    bitArrays.insert({result, bits});
  });
}

void Vectorizer::processConcatOps() {
  module.walk([&](comb::ConcatOp op) {
    mlir::Value result = op.getResult();
    BitArray concatenatedArray;
    unsigned currentBitOffset = 0;

    for (Value operand : llvm::reverse(op.getInputs())) {
      unsigned operandWidth = cast<IntegerType>(operand.getType()).getWidth();

      if (bitArrays.count(operand)) {
        BitArray &operandArray = bitArrays[operand];
        for (auto const &[bitIndex, bitInfo] : operandArray.bits) {
          concatenatedArray.bits[bitIndex + currentBitOffset] = bitInfo;
        }
      }
      currentBitOffset += operandWidth;
    }
    bitArrays.insert({result, concatenatedArray});
  });
}

void Vectorizer::applyLinearVectorization(Value oldOutputVal,
                                          Value sourceInput) {
  oldOutputVal.replaceAllUsesWith(sourceInput);
}

void Vectorizer::applyReverseVectorization(OpBuilder &builder,
                                           Value oldOutputVal,
                                           Value sourceInput) {
  builder.setInsertionPoint(*oldOutputVal.getUsers().begin());
  Location loc = sourceInput.getLoc();
  Value reversedInput =
      comb::ReverseOp::create(builder, loc, sourceInput.getType(), sourceInput);
  oldOutputVal.replaceAllUsesWith(reversedInput);
}

void Vectorizer::applyMixVectorization(OpBuilder &builder, Value oldOutputVal,
                                       Value sourceInput,
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

void Vectorizer::applyStructuralVectorization(OpBuilder &builder,
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

void Vectorizer::applyPartialVectorization(OpBuilder &builder,
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

bool Vectorizer::hasCrossBitDependencies(mlir::Value outputVal) {
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

bool Vectorizer::canVectorizeStructurally(mlir::Value output) {
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

bool Vectorizer::canApplyPartialVectorization(Value oldOutputVal) {
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

bool Vectorizer::isValidPermutation(const llvm::SmallVector<unsigned> &perm,
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

bool Vectorizer::isSafeSharedValue(mlir::Value val) {
  llvm::SmallPtrSet<mlir::Value, 8> visited;
  return isSafeSharedValue(val, visited);
}

void Vectorizer::collectLogicCone(mlir::Value val,
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

bool Vectorizer::isSafeSharedValue(
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

bool Vectorizer::areSubgraphsEquivalent(
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

mlir::Value Vectorizer::vectorizeSubgraph(
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

bool Vectorizer::isBitConstant(mlir::Value val, unsigned bitIndex,
                               bool expectedVal) {
  Operation *defOp = val.getDefiningOp();
  if (!defOp)
    return false;

  if (auto c = dyn_cast<hw::ConstantOp>(defOp)) {
    if (bitIndex < c.getValue().getBitWidth())
      return c.getValue()[bitIndex] == expectedVal;
    return false;
  }

  if (auto concat = dyn_cast<comb::ConcatOp>(defOp)) {
    unsigned current = cast<IntegerType>(val.getType()).getWidth();
    for (Value opVal : concat.getInputs()) {
      unsigned w = cast<IntegerType>(opVal.getType()).getWidth();
      current -= w;
      if (bitIndex >= current && bitIndex < current + w)
        return isBitConstant(opVal, bitIndex - current, expectedVal);
    }
    return false;
  }

  if (auto ext = dyn_cast<comb::ExtractOp>(defOp)) {
    return isBitConstant(ext.getInput(), ext.getLowBit() + bitIndex,
                         expectedVal);
  }

  if (auto andOp = dyn_cast<comb::AndOp>(defOp)) {
    bool identity = true;
    if (expectedVal == !identity) {
      for (auto input : andOp.getInputs())
        if (isBitConstant(input, bitIndex, false))
          return true;
      return false;
    } else {
      for (auto input : andOp.getInputs())
        if (!isBitConstant(input, bitIndex, true))
          return false;
      return true;
    }
  }

  if (auto orOp = dyn_cast<comb::OrOp>(defOp)) {
    bool identity = false;
    if (expectedVal == !identity) {
      for (auto input : orOp.getInputs())
        if (isBitConstant(input, bitIndex, true))
          return true;
      return false;
    } else {
      for (auto input : orOp.getInputs())
        if (!isBitConstant(input, bitIndex, false))
          return false;
      return true;
    }
  }

  return false;
}

mlir::Value Vectorizer::findBitSource(mlir::Value vectorVal, unsigned bitIndex,
                                      int depth) {
  if (auto blockArg = dyn_cast<BlockArgument>(vectorVal)) {
    if (blockArg.getType().isInteger(1) && bitIndex == 0)
      return blockArg;
    return nullptr;
  }

  Operation *op = vectorVal.getDefiningOp();
  if (!op)
    return nullptr;

  if (auto concat = dyn_cast<comb::ConcatOp>(op)) {
    unsigned currentBit = cast<IntegerType>(vectorVal.getType()).getWidth();

    for (Value operand : concat.getInputs()) {
      unsigned operandWidth = cast<IntegerType>(operand.getType()).getWidth();
      currentBit -= operandWidth;

      if (bitIndex >= currentBit && bitIndex < currentBit + operandWidth) {
        return findBitSource(operand, bitIndex - currentBit, depth + 1);
      }
    }
    return nullptr;
  }

  if (op->getNumResults() == 1 && op->getResult(0).getType().isInteger(1) &&
      bitIndex == 0) {
    return op->getResult(0);
  }

  if (auto cst = dyn_cast<hw::ConstantOp>(op)) {
    if (cst.getType().isInteger(1) && bitIndex == 0)
      return cst.getResult();
  }

  if (auto orOp = dyn_cast<comb::OrOp>(op)) {
    Value lhs = orOp.getInputs()[0];
    Value rhs = orOp.getInputs()[1];
    if (isBitConstant(lhs, bitIndex, false))
      return findBitSource(rhs, bitIndex, depth + 1);
    if (isBitConstant(rhs, bitIndex, false))
      return findBitSource(lhs, bitIndex, depth + 1);
  } else if (auto andOp = dyn_cast<comb::AndOp>(op)) {
    Value lhs = andOp.getInputs()[0];
    Value rhs = andOp.getInputs()[1];
    if (isBitConstant(lhs, bitIndex, true))
      return findBitSource(rhs, bitIndex, depth + 1);
    if (isBitConstant(rhs, bitIndex, true))
      return findBitSource(lhs, bitIndex, depth + 1);
  }

  return nullptr;
}

struct HWVectorizationPass
    : public hw::impl::HWVectorizationBase<HWVectorizationPass> {

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<circt::comb::CombDialect, circt::hw::HWDialect,
                    circt::sv::SVDialect>();
  }

  void runOnOperation() override {
    hw::HWModuleOp module = getOperation();
    Vectorizer v(module);
    v.vectorize();
  }
};

} // namespace
