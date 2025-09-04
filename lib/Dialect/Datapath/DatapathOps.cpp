//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements datapath ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Datapath/DatapathOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/KnownBits.h"

#define DEBUG_TYPE "datapath-ops"

using namespace circt;
using namespace datapath;

LogicalResult CompressOp::verify() {
  // The compressor must reduce the number of operands by at least 1 otherwise
  // it fails to perform any reduction.
  if (getNumOperands() < 3)
    return emitOpError("requires 3 or more arguments - otherwise use add");

  if (getNumResults() >= getNumOperands())
    return emitOpError("must reduce the number of operands by at least 1");

  if (getNumResults() < 2)
    return emitOpError("must produce at least 2 results");

  return success();
}

// Parser for the custom type format
// Parser for "<input-type> [<num-inputs> -> <num-outputs>]"
static ParseResult parseCompressFormat(OpAsmParser &parser,
                                       SmallVectorImpl<Type> &inputTypes,
                                       SmallVectorImpl<Type> &resultTypes) {

  int64_t inputCount, resultCount;
  Type inputElementType;

  if (parser.parseType(inputElementType) || parser.parseLSquare() ||
      parser.parseInteger(inputCount) || parser.parseArrow() ||
      parser.parseInteger(resultCount) || parser.parseRSquare())
    return failure();

  // Inputs and results have same type
  inputTypes.assign(inputCount, inputElementType);
  resultTypes.assign(resultCount, inputElementType);

  return success();
}

// Printer for "<input-type> [<num-inputs> -> <num-outputs>]"
static void printCompressFormat(OpAsmPrinter &printer, Operation *op,
                                TypeRange inputTypes, TypeRange resultTypes) {

  printer << inputTypes[0] << " [" << inputTypes.size() << " -> "
          << resultTypes.size() << "]";
}

//===----------------------------------------------------------------------===//
// Compressor Tree Logic.
//===----------------------------------------------------------------------===//

// Construct a full adder for three 1-bit inputs.
std::pair<CompressorBit, CompressorBit>
CompressorTree::fullAdderWithDelay(OpBuilder &builder, CompressorBit a,
                                   CompressorBit b, CompressorBit c) {

  auto aXorB = builder.createOrFold<comb::XorOp>(loc, a.val, b.val, true);
  Value sumVal = builder.createOrFold<comb::XorOp>(loc, aXorB, c.val, true);

  auto carryVal = builder.createOrFold<comb::OrOp>(
      loc,
      ArrayRef<Value>{
          builder.createOrFold<comb::AndOp>(loc, a.val, b.val, true),
          builder.createOrFold<comb::AndOp>(loc, aXorB, c.val, true)},
      true);

  auto sumDelay = std::max(std::max(a.delay, b.delay) + 1, c.delay) + 1;
  auto carryDelay = sumDelay + 1;

  CompressorBit sum = {sumVal, sumDelay};
  CompressorBit carry = {carryVal, carryDelay};
  std::pair<CompressorBit, CompressorBit> fa{sum, carry};
  ++numFullAdders;
  return fa;
}

// Construct a half adder for two 1-bit inputs.
std::pair<CompressorBit, CompressorBit>
CompressorTree::halfAdderWithDelay(OpBuilder &builder, CompressorBit a,
                                   CompressorBit b) {
  auto sumVal = builder.createOrFold<comb::XorOp>(loc, a.val, b.val, true);
  auto carryVal = builder.createOrFold<comb::AndOp>(loc, a.val, b.val, true);

  auto sumDelay = std::max(a.delay, b.delay) + 1;
  auto carryDelay = sumDelay;

  CompressorBit sum = {sumVal, sumDelay};
  CompressorBit carry = {carryVal, carryDelay};
  std::pair<CompressorBit, CompressorBit> ha{sum, carry};
  return ha;
}

// Map input rows to column representation
CompressorTree::CompressorTree(const SmallVector<SmallVector<Value>> &addends,
                               Location loc)
    : originalAddends(addends), usingTiming(false), numStages(0),
      numFullAdders(0), loc(loc) {
  assert(addends.size() > 2);
  // Number of bits in a row == bitwidth of input addends
  // Compressors will be formed of uniform bitwidth addends
  width = addends[0].size();
  SmallVector<SmallVector<CompressorBit>> initColumns(width);
  columns = initColumns;
  // Convert addends rows to columns
  // Known bits analysis constructs a minimal array - skipping zeros
  for (auto row : addends) {
    for (size_t i = 0; i < width; ++i) {
      CompressorBit bit = {row[i], 0};
      // TODO: Fold Constant 1s
      auto knownBits = comb::computeKnownBits(bit.val);
      if (knownBits.isZero())
        continue;
      // Add non-zero bit to the column
      columns[i].push_back(bit);
    }
  }
}

// Update the input delays based on longest path analysis
void CompressorTree::withInputDelays(
    const SmallVector<SmallVector<int64_t>> &inputDelays) {
  assert(inputDelays.size() == originalAddends.size() &&
         "Input delays must match number of addends");
  for (size_t i = 0; i < inputDelays.size(); ++i) {
    assert(inputDelays[i].size() == width &&
           "Input delays must match bitwidth of addends");
    for (size_t j = 0; j < width; ++j) {
      // Find the corresponding bit in the column and update its delay
      auto it = std::find_if(columns[j].begin(), columns[j].end(),
                             [val = originalAddends[i][j]](const auto &bit) {
                               return bit.val == val;
                             });
      if (it != columns[j].end())
        it->delay = inputDelays[i][j];
    }
  }
  usingTiming = true;
}

size_t CompressorTree::getMaxHeight() const {
  size_t maxSize = 0;
  for (const auto &column : columns)
    maxSize = std::max(maxSize, column.size());

  return maxSize;
}

// Use Dadda's ALAP alogrithm to determine the target height of the next stage
// https://en.wikipedia.org/wiki/Dadda_multiplier
size_t CompressorTree::getNextStageTargetHeight() const {
  auto maxHeight = getMaxHeight();
  size_t m_prev = 2;
  while (true) {
    size_t m = static_cast<size_t>(std::floor(1.5 * m_prev));
    if (m >= maxHeight)
      return m_prev;
    m_prev = m;
  }
}

// Convert back to a concatenated addend representation
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

  if (maxHeight <= targetHeight)
    return columnsToAddends(builder, targetHeight);

  if (usingTiming)
    return compressUsingTiming(builder, targetHeight);
  else
    return compressWithoutTiming(builder, targetHeight);
}

// Perform recursive compression using timing information until reduced to the
// target height - this currently uses Dadda's algorithm and timing driven
// signal selection
// TODO: Dadda's algorithm is redundant here since it assumes uniform arrival so
// need to implement a more timing driven approach
SmallVector<Value> CompressorTree::compressUsingTiming(OpBuilder &builder,
                                                       size_t targetHeight) {
  while (getMaxHeight() > targetHeight) {
    dump();
    // Increment the number of reduction stages for debugging/reporting
    ++numStages;

    // Use Dadda's algorithm to compute next stage height
    auto targetStageHeight = getNextStageTargetHeight();
    // Initialize empty newColumns
    SmallVector<SmallVector<CompressorBit>> newColumns(width);

    for (size_t i = 0; i < width; ++i) {
      auto col = columns[i];

      // Sort the column by arrival time - fastest at the end
      std::stable_sort(
          col.begin(), col.end(),
          [](const auto &a, const auto &b) { return a.delay > b.delay; });
      // Only compress to reach the target stage height - Dadda's Algorithm
      while (col.size() + newColumns[i].size() > targetStageHeight) {
        if (col.size() < 2) {
          llvm::errs() << "CompressorTree: Not enough bits in column " << i
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
          // bit2 can arrive 1 delay unit after bit0 and bit1 without delaying
          // the full-adder
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
          auto [sum, carry] = fullAdderWithDelay(builder, bit0, bit1, bit2);

          newColumns[i].push_back(sum);
          if (i + 1 < newColumns.size())
            newColumns[i + 1].push_back(carry);
        } else {
          // Apply a half adder to bit0 and bit1
          auto [sum, carry] = halfAdderWithDelay(builder, bit0, bit1);

          newColumns[i].push_back(sum);
          if (i + 1 < newColumns.size())
            newColumns[i + 1].push_back(carry);
        }
      }

      // Pass through remaining bits
      for (auto bit : col)
        newColumns[i].push_back(bit);
    }

    // Compute another stage of reduction
    columns = newColumns;
  }
  dump();
  return columnsToAddends(builder, targetHeight);
}

// Compress using a greedy algorithm that just picks the top three bits
// without any timing consideration - Wallace Tree Reduction
// See https://en.wikipedia.org/wiki/Wallace_tree
SmallVector<Value> CompressorTree::compressWithoutTiming(OpBuilder &builder,
                                                         size_t targetHeight) {

  // Continue reduction until we have only two rows. The length of
  // `addends` is reduced by 1/3 in each iteration.
  while (getMaxHeight() > targetHeight) {
    SmallVector<SmallVector<CompressorBit>> newColumns(width);
    dump();
    // Increment the number of reduction stages for debugging/reporting
    ++numStages;
    // Take three rows at a time and reduce to two rows(sum and carry).
    for (size_t i = 0; i < width; ++i) {
      auto col = columns[i];
      while (col.size() >= 3) {
        auto bit0 = col.pop_back_val();
        auto bit1 = col.pop_back_val();
        auto bit2 = col.pop_back_val();

        // If we have an additional bit we can apply a full adder
        auto [sum, carry] = fullAdderWithDelay(builder, bit0, bit1, bit2);

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
  dump();
  return columnsToAddends(builder, targetHeight);
}

void CompressorTree::dump() const {
  LLVM_DEBUG({
    llvm::dbgs() << "Compressor Tree: Height = " << getMaxHeight()
                 << ", Number of FA = " << numFullAdders
                 << ", Number of Stages = " << numStages;
    if (usingTiming)
      llvm::dbgs() << ", Next Stage Target = " << getNextStageTargetHeight();

    llvm::dbgs() << "\n";
    // Print column headers
    llvm::dbgs() << std::string(9, ' ');
    for (size_t j = width; j > 0; --j) {
      if (j < width)
        llvm::dbgs() << " ";
      llvm::dbgs() << llvm::format("%02d", j - 1);
    }
    llvm::dbgs() << "\n"
                 << std::string(9, ' ') << std::string(width * 3, '-') << "\n";

    for (size_t i = 0; i < getMaxHeight(); ++i) {
      llvm::dbgs() << "  [" << llvm::format("%02d", i) << "]: [";
      for (size_t j = width; j > 0; --j) {
        if (j < width)
          llvm::dbgs() << " ";
        if (i < columns[j - 1].size())
          llvm::dbgs() << llvm::format(
              "%02d",
              columns[j - 1][i].delay); // Assumes CompressorBit has operator
        else
          llvm::dbgs() << "  ";
      }
      llvm::dbgs() << "]\n";
    }
  });
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/Datapath/Datapath.cpp.inc"
