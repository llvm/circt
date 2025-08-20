//===- CombOps.cpp - Implement the Comb operations ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements combinational ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/KnownBits.h"
#include <algorithm>

#define DEBUG_TYPE "comb-ops"

using namespace circt;
using namespace comb;

Value comb::createZExt(OpBuilder &builder, Location loc, Value value,
                       unsigned targetWidth) {
  assert(value.getType().isSignlessInteger());
  auto inputWidth = value.getType().getIntOrFloatBitWidth();
  assert(inputWidth <= targetWidth);

  // Nothing to do if the width already matches.
  if (inputWidth == targetWidth)
    return value;

  // Create a zero constant for the upper bits.
  auto zeros = hw::ConstantOp::create(
      builder, loc, builder.getIntegerType(targetWidth - inputWidth), 0);
  return builder.createOrFold<ConcatOp>(loc, zeros, value);
}

/// Create a sign extension operation from a value of integer type to an equal
/// or larger integer type.
Value comb::createOrFoldSExt(Location loc, Value value, Type destTy,
                             OpBuilder &builder) {
  IntegerType valueType = dyn_cast<IntegerType>(value.getType());
  assert(valueType && isa<IntegerType>(destTy) &&
         valueType.getWidth() <= destTy.getIntOrFloatBitWidth() &&
         valueType.getWidth() != 0 && "invalid sext operands");
  // If already the right size, we are done.
  if (valueType == destTy)
    return value;

  // sext is concat with a replicate of the sign bits and the bottom part.
  auto signBit =
      builder.createOrFold<ExtractOp>(loc, value, valueType.getWidth() - 1, 1);
  auto signBits = builder.createOrFold<ReplicateOp>(
      loc, signBit, destTy.getIntOrFloatBitWidth() - valueType.getWidth());
  return builder.createOrFold<ConcatOp>(loc, signBits, value);
}

Value comb::createOrFoldSExt(Value value, Type destTy,
                             ImplicitLocOpBuilder &builder) {
  return createOrFoldSExt(builder.getLoc(), value, destTy, builder);
}

Value comb::createOrFoldNot(Location loc, Value value, OpBuilder &builder,
                            bool twoState) {
  auto allOnes = hw::ConstantOp::create(builder, loc, value.getType(), -1);
  return builder.createOrFold<XorOp>(loc, value, allOnes, twoState);
}

Value comb::createOrFoldNot(Value value, ImplicitLocOpBuilder &builder,
                            bool twoState) {
  return createOrFoldNot(builder.getLoc(), value, builder, twoState);
}

// Extract individual bits from a value
void comb::extractBits(OpBuilder &builder, Value val,
                       SmallVectorImpl<Value> &bits) {
  assert(val.getType().isInteger() && "expected integer");
  auto width = val.getType().getIntOrFloatBitWidth();
  bits.reserve(width);

  // Check if we can reuse concat operands
  if (auto concat = val.getDefiningOp<comb::ConcatOp>()) {
    if (concat.getNumOperands() == width &&
        llvm::all_of(concat.getOperandTypes(), [](Type type) {
          return type.getIntOrFloatBitWidth() == 1;
        })) {
      // Reverse the operands to match the bit order
      bits.append(std::make_reverse_iterator(concat.getOperands().end()),
                  std::make_reverse_iterator(concat.getOperands().begin()));
      return;
    }
  }

  // Extract individual bits
  for (int64_t i = 0; i < width; ++i)
    bits.push_back(
        builder.createOrFold<comb::ExtractOp>(val.getLoc(), val, i, 1));
}

// Construct a mux tree for given leaf nodes. `selectors` is the selector for
// each level of the tree. Currently the selector is tested from MSB to LSB.
Value comb::constructMuxTree(OpBuilder &builder, Location loc,
                             ArrayRef<Value> selectors,
                             ArrayRef<Value> leafNodes,
                             Value outOfBoundsValue) {
  // Recursive helper function to construct the mux tree
  std::function<Value(size_t, size_t)> constructTreeHelper =
      [&](size_t id, size_t level) -> Value {
    // Base case: at the lowest level, return the result
    if (level == 0) {
      // Return the result for the given index. If the index is out of bounds,
      // return the out-of-bound value.
      return id < leafNodes.size() ? leafNodes[id] : outOfBoundsValue;
    }

    auto selector = selectors[level - 1];

    // Recursive case: create muxes for true and false branches
    auto trueVal = constructTreeHelper(2 * id + 1, level - 1);
    auto falseVal = constructTreeHelper(2 * id, level - 1);

    // Combine the results with a mux
    return builder.createOrFold<comb::MuxOp>(loc, selector, trueVal, falseVal);
  };

  return constructTreeHelper(0, llvm::Log2_64_Ceil(leafNodes.size()));
}

Value comb::createDynamicExtract(OpBuilder &builder, Location loc, Value value,
                                 Value offset, unsigned width) {
  assert(value.getType().isSignlessInteger());
  auto valueWidth = value.getType().getIntOrFloatBitWidth();
  assert(width <= valueWidth);

  // Handle the special case where the offset is constant.
  APInt constOffset;
  if (matchPattern(offset, mlir::m_ConstantInt(&constOffset)))
    if (constOffset.getActiveBits() < 32)
      return builder.createOrFold<comb::ExtractOp>(
          loc, value, constOffset.getZExtValue(), width);

  // Zero-extend the offset, shift the value down, and extract the requested
  // number of bits.
  offset = createZExt(builder, loc, offset, valueWidth);
  value = builder.createOrFold<comb::ShrUOp>(loc, value, offset);
  return builder.createOrFold<comb::ExtractOp>(loc, value, 0, width);
}

Value comb::createDynamicInject(OpBuilder &builder, Location loc, Value value,
                                Value offset, Value replacement,
                                bool twoState) {
  assert(value.getType().isSignlessInteger());
  assert(replacement.getType().isSignlessInteger());
  auto largeWidth = value.getType().getIntOrFloatBitWidth();
  auto smallWidth = replacement.getType().getIntOrFloatBitWidth();
  assert(smallWidth <= largeWidth);

  // If we're inserting a zero-width value there's nothing to do.
  if (smallWidth == 0)
    return value;

  // Handle the special case where the offset is constant.
  APInt constOffset;
  if (matchPattern(offset, mlir::m_ConstantInt(&constOffset)))
    if (constOffset.getActiveBits() < 32)
      return createInject(builder, loc, value, constOffset.getZExtValue(),
                          replacement);

  // Zero-extend the offset and clear the value bits we are replacing.
  offset = createZExt(builder, loc, offset, largeWidth);
  Value mask = hw::ConstantOp::create(
      builder, loc, APInt::getLowBitsSet(largeWidth, smallWidth));
  mask = builder.createOrFold<comb::ShlOp>(loc, mask, offset);
  mask = createOrFoldNot(loc, mask, builder, true);
  value = builder.createOrFold<comb::AndOp>(loc, value, mask, twoState);

  // Zero-extend the replacement value, shift it up to the offset, and merge it
  // with the value that has the corresponding bits cleared.
  replacement = createZExt(builder, loc, replacement, largeWidth);
  replacement = builder.createOrFold<comb::ShlOp>(loc, replacement, offset);
  return builder.createOrFold<comb::OrOp>(loc, value, replacement, twoState);
}

Value comb::createInject(OpBuilder &builder, Location loc, Value value,
                         unsigned offset, Value replacement) {
  assert(value.getType().isSignlessInteger());
  assert(replacement.getType().isSignlessInteger());
  auto largeWidth = value.getType().getIntOrFloatBitWidth();
  auto smallWidth = replacement.getType().getIntOrFloatBitWidth();
  assert(smallWidth <= largeWidth);

  // If the offset is outside the value there's nothing to do.
  if (offset >= largeWidth)
    return value;

  // If we're inserting a zero-width value there's nothing to do.
  if (smallWidth == 0)
    return value;

  // Assemble the pieces of the injection as everything below the offset, the
  // replacement value, and everything above the replacement value.
  SmallVector<Value, 3> fragments;
  auto end = offset + smallWidth;
  if (end < largeWidth)
    fragments.push_back(
        comb::ExtractOp::create(builder, loc, value, end, largeWidth - end));
  if (end <= largeWidth)
    fragments.push_back(replacement);
  else
    fragments.push_back(comb::ExtractOp::create(builder, loc, replacement, 0,
                                                largeWidth - offset));
  if (offset > 0)
    fragments.push_back(
        comb::ExtractOp::create(builder, loc, value, 0, offset));
  return builder.createOrFold<comb::ConcatOp>(loc, fragments);
}

// Construct a full adder for three 1-bit inputs.
std::pair<Value, Value> comb::fullAdder(OpBuilder &builder, Location loc,
                                        Value a, Value b, Value c) {
  auto aXorB = builder.createOrFold<comb::XorOp>(loc, a, b, true);
  Value sum = builder.createOrFold<comb::XorOp>(loc, aXorB, c, true);

  auto carry = builder.createOrFold<comb::OrOp>(
      loc,
      ArrayRef<Value>{builder.createOrFold<comb::AndOp>(loc, a, b, true),
                      builder.createOrFold<comb::AndOp>(loc, aXorB, c, true)},
      true);

  return {sum, carry};
}

// Construct a full adder for three 1-bit inputs.
std::pair<CompressorBit, CompressorBit>
comb::fullAdderWithDelay(OpBuilder &builder, Location loc, CompressorBit a,
                         CompressorBit b, CompressorBit c) {
  auto [sum_val, carry_val] =
      comb::fullAdder(builder, loc, a.val, b.val, c.val);

  auto sum_delay = std::max(std::max(a.delay, b.delay) + 1, c.delay) + 1;
  auto carry_delay = sum_delay + 1;

  CompressorBit sum = {sum_val, sum_delay};
  CompressorBit carry = {carry_val, carry_delay};
  std::pair<CompressorBit, CompressorBit> fa{sum, carry};
  return fa;
}

// Construct a full adder for three 1-bit inputs.
std::pair<CompressorBit, CompressorBit>
comb::halfAdderWithDelay(OpBuilder &builder, Location loc, CompressorBit a,
                         CompressorBit b) {
  auto sum_val = builder.createOrFold<comb::XorOp>(loc, a.val, b.val, true);
  auto carry_val = builder.createOrFold<comb::AndOp>(loc, a.val, b.val, true);

  auto sum_delay = std::max(a.delay, b.delay) + 1;
  auto carry_delay = sum_delay;

  CompressorBit sum = {sum_val, sum_delay};
  CompressorBit carry = {carry_val, carry_delay};
  std::pair<CompressorBit, CompressorBit> ha{sum, carry};
  return ha;
}

CompressorTree::CompressorTree(const SmallVector<SmallVector<Value>> &addends,
                               Location loc)
    : originalAddends(addends), loc(loc), numStages(0), numFullAdders(0),
      usingTiming(true) {
  assert(addends.size() > 2);
  // Number of bits in a row == bitwidth of input addends
  // Compressors will be formed of uniform bitwidth addends
  width = addends[0].size();

  // Known bits analysis constructs a minimal array
  for (auto row : addends) {
    for (size_t i = 0; i < width; ++i) {
      // TODO - incorporate non-zero arrival delays
      CompressorBit bit = {row[i], 0};
      auto knownBits = computeKnownBits(bit.val);
      if (knownBits.isZero())
        continue;
      if (columns.size() < width) {
        SmallVector<CompressorBit> col{bit};
        columns.push_back(col);
      } else {
        columns[i].push_back(bit);
      }
    }
  }
}

size_t CompressorTree::getMaxHeight() const {
  size_t maxSize = 0;
  for (const auto &column : columns) {
    maxSize = std::max(maxSize, column.size());
  }
  return maxSize;
}

size_t CompressorTree::getNextStageTargetHeight() const {
  // TODO - improve computation of Dadda sequence
  auto maxHeight = getMaxHeight();
  SmallVector<size_t> daddaSequence{2, 3, 4, 6, 9, 13, 19, 28, 42};
  for (size_t i = 0; i < daddaSequence.size() - 1; ++i) {
    if (daddaSequence[i + 1] >= maxHeight)
      return daddaSequence[i];
  }

  return daddaSequence.back();
}

SmallVector<Value> CompressorTree::columnsToAddends(OpBuilder &builder,
                                                    size_t targetHeight) {
  SmallVector<Value> addend;
  SmallVector<Value> addends;
  auto falseValue = hw::ConstantOp::create(builder, loc, APInt(1, 0));
  for (size_t i = 0; i < targetHeight; ++i) {
    // Pad with zeros
    if (i >= getMaxHeight()) {
      addends.push_back(hw::ConstantOp::create(builder, loc, APInt(width, 0)));
      continue;
    }
    // Otherwise populate a addend formed from a concatenation
    for (size_t j = 0; j < width; ++j) {
      if (i < columns[j].size())
        addend.push_back(columns[j][i].val);
      else {
        addend.push_back(falseValue);
      }
    }
    std::reverse(addend.begin(), addend.end());
    addends.push_back(comb::ConcatOp::create(builder, loc, addend));
    addend.clear();
  }
  return addends;
}

// Perform recursive compression until reduced to the target height
SmallVector<Value> CompressorTree::compressToHeight(OpBuilder &builder,
                                                    size_t targetHeight) {

  auto maxHeight = getMaxHeight();
  dump();

  if (maxHeight <= targetHeight)
    return columnsToAddends(builder, targetHeight);

  if (usingTiming)
    return compressUsingTiming(builder, targetHeight);
  else
    return compressWithoutTiming(builder, targetHeight);
}

// Perform recursive compression using timing information until reduced to the
// target height - this currently uses Dadda's algorithm
SmallVector<Value> CompressorTree::compressUsingTiming(OpBuilder &builder,
                                                       size_t targetHeight) {

  dump();
  auto maxHeight = getMaxHeight();
  auto targetStageHeight = getNextStageTargetHeight();

  // TODO: Refactor to avoid recursion
  if (maxHeight <= targetHeight)
    return columnsToAddends(builder, targetHeight);

  // Increment the number of reduction stages for debugging/reporting
  ++numStages;
  // Initialize empty newColumns
  SmallVector<SmallVector<CompressorBit>> newColumns(width);
  // newColumns.reserve(width);
  // SmallVector<CompressorBit> empty{};
  // while (newColumns.size() < width)
  // newColumns.push_back(empty);

  for (size_t i = 0; i < width; ++i) {
    auto col = columns[i];

    // Sort the column by arrival time - fastest at the end
    std::sort(col.begin(), col.end(),
              [](const auto &a, const auto &b) { return a.delay > b.delay; });
    // Only compress to reach the target stage height - Dadda's Algorithm
    while (col.size() + newColumns[i].size() > targetStageHeight) {
      if (col.size() < 2) {
        llvm::dbgs() << "CompressorTree: Not enough bits in column " << i
                     << " to compress further.\n New Columns size: "
                     << newColumns[i].size()
                     << ", Current Column size: " << col.size() << "\n";
      }
      assert(col.size() >= 2 &&
             "Expected at least two bits in compressor column");
      auto bit0 = col.pop_back_val();
      auto bit1 = col.pop_back_val();

      // If we have an additional bit we can apply a full adder
      if (col.size() >= 1) {
        // bit2 can arrive 1 delay unit after bit0 and bit1 without delaying the
        // full-adder
        auto targetDelay = std::max(bit0.delay, bit1.delay) + 1;
        CompressorBit bit2;

        // Find the third bit of the full-adder that satisfies the delay
        // constraint
        auto it = std::find_if(col.begin(), col.end(),
                               [targetDelay](const auto &pair) {
                                 return pair.delay <= targetDelay;
                               });

        if (it != col.end()) {
          bit2 = *it;
          col.erase(it);
        } else {
          // If no bit satisfies the delay constraint pick the fastest one
          bit2 = col.pop_back_val();
        }
        auto [sum, carry] = fullAdderWithDelay(builder, loc, bit0, bit1, bit2);
        ++numFullAdders;

        newColumns[i].push_back(sum);
        if (i + 1 < newColumns.size())
          newColumns[i + 1].push_back(carry);
      } else {
        // Apply a half adder to bit0 and bit1
        auto [sum, carry] = halfAdderWithDelay(builder, loc, bit0, bit1);

        newColumns[i].push_back(sum);
        if (i + 1 < newColumns.size())
          newColumns[i + 1].push_back(carry);
      }
    }

    // Pass through remaining columns
    for (auto bit : col)
      newColumns[i].push_back(bit);
  }

  // Compute another stage of reduction
  columns = newColumns;
  // Check that we reduced the maximum height
  assert(getMaxHeight() < maxHeight);
  return compressUsingTiming(builder, targetHeight);
}

SmallVector<Value> CompressorTree::compressWithoutTiming(OpBuilder &builder,
                                                         size_t targetHeight) {
  auto falseValue = hw::ConstantOp::create(builder, loc, APInt(1, 0));
  SmallVector<SmallVector<CompressorBit>> newColumns;
  newColumns.reserve(width);
  // Continue reduction until we have only two rows. The length of
  // `addends` is reduced by 1/3 in each iteration.
  while (getMaxHeight() > targetHeight) {
    newColumns.clear();
    // Take three rows at a time and reduce to two rows(sum and carry).
    for (size_t i = 0; i < width; ++i) {
      auto col = columns[i];
      while (col.size() >= 3) {
        auto bit0 = col.pop_back_val();
        auto bit1 = col.pop_back_val();
        auto bit2 = col.pop_back_val();

        // If we have an additional bit we can apply a full adder
        auto [sum, carry] = fullAdderWithDelay(builder, loc, bit0, bit1, bit2);
        ++numFullAdders;

        newColumns[i].push_back(sum);
        if (i + 1 < width)
          newColumns[i + 1].push_back(carry);
      }
      // Pass through remaining bits
      for (auto bit : col)
        newColumns[i].push_back(bit);
    }

    std::swap(newColumns, columns);
  }

  assert(getMaxHeight() <= targetHeight);
  return columnsToAddends(builder, targetHeight);
}

void CompressorTree::dump() const {
  LLVM_DEBUG({
    llvm::dbgs() << "Compressor Tree: Height = " << getMaxHeight()
                 << ", Number of Stages = " << numStages
                 << ", Next Stage Target = " << getNextStageTargetHeight()
                 << ", Number of FA = " << numFullAdders << "\n";
    // Print column headers
    llvm::dbgs() << std::string(9, ' ');
    for (int j = width - 1; j >= 0; --j) {
      if (j < width - 1)
        llvm::dbgs() << " ";
      llvm::dbgs() << llvm::format("%02d", j);
    }
    llvm::dbgs() << "\n"
                 << std::string(9, ' ') << std::string(width * 3, '-') << "\n";

    for (size_t i = 0; i < getMaxHeight(); ++i) {
      llvm::dbgs() << "  [" << llvm::format("%02d", i) << "]: [";
      for (int j = width - 1; j >= 0; --j) {
        if (j < width - 1)
          llvm::dbgs() << " ";
        if (i < columns[j].size())
          llvm::dbgs() << llvm::format(
              "%02d",
              columns[j][i].delay); // Assumes CompressorBit has operator
        else
          llvm::dbgs() << "  ";
      }
      llvm::dbgs() << "]\n";
    }
  });
}
//===----------------------------------------------------------------------===//
// ICmpOp
//===----------------------------------------------------------------------===//

ICmpPredicate ICmpOp::getFlippedPredicate(ICmpPredicate predicate) {
  switch (predicate) {
  case ICmpPredicate::eq:
    return ICmpPredicate::eq;
  case ICmpPredicate::ne:
    return ICmpPredicate::ne;
  case ICmpPredicate::slt:
    return ICmpPredicate::sgt;
  case ICmpPredicate::sle:
    return ICmpPredicate::sge;
  case ICmpPredicate::sgt:
    return ICmpPredicate::slt;
  case ICmpPredicate::sge:
    return ICmpPredicate::sle;
  case ICmpPredicate::ult:
    return ICmpPredicate::ugt;
  case ICmpPredicate::ule:
    return ICmpPredicate::uge;
  case ICmpPredicate::ugt:
    return ICmpPredicate::ult;
  case ICmpPredicate::uge:
    return ICmpPredicate::ule;
  case ICmpPredicate::ceq:
    return ICmpPredicate::ceq;
  case ICmpPredicate::cne:
    return ICmpPredicate::cne;
  case ICmpPredicate::weq:
    return ICmpPredicate::weq;
  case ICmpPredicate::wne:
    return ICmpPredicate::wne;
  }
  llvm_unreachable("unknown comparison predicate");
}

bool ICmpOp::isPredicateSigned(ICmpPredicate predicate) {
  switch (predicate) {
  case ICmpPredicate::ult:
  case ICmpPredicate::ugt:
  case ICmpPredicate::ule:
  case ICmpPredicate::uge:
  case ICmpPredicate::ne:
  case ICmpPredicate::eq:
  case ICmpPredicate::cne:
  case ICmpPredicate::ceq:
  case ICmpPredicate::wne:
  case ICmpPredicate::weq:
    return false;
  case ICmpPredicate::slt:
  case ICmpPredicate::sgt:
  case ICmpPredicate::sle:
  case ICmpPredicate::sge:
    return true;
  }
  llvm_unreachable("unknown comparison predicate");
}

/// Returns the predicate for a logically negated comparison, e.g. mapping
/// EQ => NE and SLE => SGT.
ICmpPredicate ICmpOp::getNegatedPredicate(ICmpPredicate predicate) {
  switch (predicate) {
  case ICmpPredicate::eq:
    return ICmpPredicate::ne;
  case ICmpPredicate::ne:
    return ICmpPredicate::eq;
  case ICmpPredicate::slt:
    return ICmpPredicate::sge;
  case ICmpPredicate::sle:
    return ICmpPredicate::sgt;
  case ICmpPredicate::sgt:
    return ICmpPredicate::sle;
  case ICmpPredicate::sge:
    return ICmpPredicate::slt;
  case ICmpPredicate::ult:
    return ICmpPredicate::uge;
  case ICmpPredicate::ule:
    return ICmpPredicate::ugt;
  case ICmpPredicate::ugt:
    return ICmpPredicate::ule;
  case ICmpPredicate::uge:
    return ICmpPredicate::ult;
  case ICmpPredicate::ceq:
    return ICmpPredicate::cne;
  case ICmpPredicate::cne:
    return ICmpPredicate::ceq;
  case ICmpPredicate::weq:
    return ICmpPredicate::wne;
  case ICmpPredicate::wne:
    return ICmpPredicate::weq;
  }
  llvm_unreachable("unknown comparison predicate");
}

/// Return true if this is an equality test with -1, which is a "reduction
/// and" operation in Verilog.
bool ICmpOp::isEqualAllOnes() {
  if (getPredicate() != ICmpPredicate::eq)
    return false;

  if (auto op1 =
          dyn_cast_or_null<hw::ConstantOp>(getOperand(1).getDefiningOp()))
    return op1.getValue().isAllOnes();
  return false;
}

/// Return true if this is a not equal test with 0, which is a "reduction
/// or" operation in Verilog.
bool ICmpOp::isNotEqualZero() {
  if (getPredicate() != ICmpPredicate::ne)
    return false;

  if (auto op1 =
          dyn_cast_or_null<hw::ConstantOp>(getOperand(1).getDefiningOp()))
    return op1.getValue().isZero();
  return false;
}

//===----------------------------------------------------------------------===//
// Unary Operations
//===----------------------------------------------------------------------===//

LogicalResult ReplicateOp::verify() {
  // The source must be equal or smaller than the dest type, and an even
  // multiple of it.  Both are already known to be signless integers.
  auto srcWidth = cast<IntegerType>(getOperand().getType()).getWidth();
  auto dstWidth = cast<IntegerType>(getType()).getWidth();
  if (srcWidth == 0)
    return emitOpError("replicate does not take zero bit integer");

  if (srcWidth > dstWidth)
    return emitOpError("replicate cannot shrink bitwidth of operand"),
           failure();

  if (dstWidth % srcWidth)
    return emitOpError("replicate must produce integer multiple of operand"),
           failure();

  return success();
}

//===----------------------------------------------------------------------===//
// Variadic operations
//===----------------------------------------------------------------------===//

static LogicalResult verifyUTBinOp(Operation *op) {
  if (op->getOperands().empty())
    return op->emitOpError("requires 1 or more args");
  return success();
}

LogicalResult AddOp::verify() { return verifyUTBinOp(*this); }

LogicalResult MulOp::verify() { return verifyUTBinOp(*this); }

LogicalResult AndOp::verify() { return verifyUTBinOp(*this); }

LogicalResult OrOp::verify() { return verifyUTBinOp(*this); }

LogicalResult XorOp::verify() { return verifyUTBinOp(*this); }

/// Return true if this is a two operand xor with an all ones constant as
/// its RHS operand.
bool XorOp::isBinaryNot() {
  if (getNumOperands() != 2)
    return false;
  if (auto cst = getOperand(1).getDefiningOp<hw::ConstantOp>())
    if (cst.getValue().isAllOnes())
      return true;
  return false;
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

static unsigned getTotalWidth(ValueRange inputs) {
  unsigned resultWidth = 0;
  for (auto input : inputs) {
    resultWidth += hw::type_cast<IntegerType>(input.getType()).getWidth();
  }
  return resultWidth;
}

void ConcatOp::build(OpBuilder &builder, OperationState &result, Value hd,
                     ValueRange tl) {
  result.addOperands(ValueRange{hd});
  result.addOperands(tl);
  unsigned hdWidth = cast<IntegerType>(hd.getType()).getWidth();
  result.addTypes(builder.getIntegerType(getTotalWidth(tl) + hdWidth));
}

LogicalResult ConcatOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &results) {
  unsigned resultWidth = getTotalWidth(operands);
  results.push_back(IntegerType::get(context, resultWidth));
  return success();
}

//===----------------------------------------------------------------------===//
// ReverseOp
//===----------------------------------------------------------------------===//

// Folding of ReverseOp: if the input is constant, compute the reverse at
// compile time.
OpFoldResult comb::ReverseOp::fold(FoldAdaptor adaptor) {
  // Try to cast the input attribute to an IntegerAttr.
  auto cstInput = llvm::dyn_cast_or_null<mlir::IntegerAttr>(adaptor.getInput());
  if (!cstInput)
    return {};

  APInt val = cstInput.getValue();
  APInt reversedVal = val.reverseBits();

  return mlir::IntegerAttr::get(getType(), reversedVal);
}

namespace {
struct ReverseOfReverse : public OpRewritePattern<comb::ReverseOp> {
  using OpRewritePattern<comb::ReverseOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(comb::ReverseOp op,
                                PatternRewriter &rewriter) const override {
    auto inputOp = op.getInput().getDefiningOp<comb::ReverseOp>();
    if (!inputOp)
      return failure();

    rewriter.replaceOp(op, inputOp.getInput());
    return success();
  }
};
} // namespace

void comb::ReverseOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  results.add<ReverseOfReverse>(context);
}

//===----------------------------------------------------------------------===//
// Other Operations
//===----------------------------------------------------------------------===//

LogicalResult ExtractOp::verify() {
  unsigned srcWidth = cast<IntegerType>(getInput().getType()).getWidth();
  unsigned dstWidth = cast<IntegerType>(getType()).getWidth();
  if (getLowBit() >= srcWidth || srcWidth - getLowBit() < dstWidth)
    return emitOpError("from bit too large for input"), failure();

  return success();
}

LogicalResult TruthTableOp::verify() {
  size_t numInputs = getInputs().size();
  if (numInputs >= sizeof(size_t) * 8)
    return emitOpError("Truth tables support a maximum of ")
           << sizeof(size_t) * 8 - 1 << " inputs on your platform";

  ArrayAttr table = getLookupTable();
  if (table.size() != (1ull << numInputs))
    return emitOpError("Expected lookup table of 2^n length");
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/Comb/Comb.cpp.inc"
