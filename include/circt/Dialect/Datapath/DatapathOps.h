//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_DATAPATH_DATAPATHOPS_H
#define CIRCT_DIALECT_DATAPATH_DATAPATHOPS_H

#include "mlir/IR/OpImplementation.h"

#include "circt/Dialect/Datapath/DatapathDialect.h"
#include "circt/Dialect/HW/HWTypes.h"

#define GET_OP_CLASSES
#include "circt/Dialect/Datapath/Datapath.h.inc"

namespace circt {
namespace datapath {

// A pair used to track compressor bits
struct CompressorBit {
  Value val;
  size_t delay;
};

// A general compressor tree interface based around a column-wise
// representation. Contains alternative compression algorithms.
class CompressorTree {
public:
  // Constructor takes addends as input and converts to column representation
  CompressorTree(const SmallVector<SmallVector<Value>> &addends, Location loc);

  // Get the number of columns (bit positions)
  size_t getWidth() const { return columns.size(); }

  // Get the maximum height of the addend array
  size_t getMaxHeight() const;

  // Get the target height of next stage
  size_t getNextStageTargetHeight() const;

  // Update the input delays based on longest path analysis
  void withInputDelays(const SmallVector<SmallVector<int64_t>> inputDelays);

  // Apply a compression step (reduce columns with >2 bits using compressors)
  SmallVector<Value> compressToHeight(OpBuilder &builder, size_t targetHeight);

  // Debug: print the tree structure
  void dump() const;

private:
  // Original addends representation as bitvectors (kept for reference)
  SmallVector<SmallVector<Value>> originalAddends;

  // Column-wise bit storage - columns[i] contains all bits at bit position i
  SmallVector<SmallVector<CompressorBit>> columns;

  // Whether to use a timing driven compression algorithm
  // If true, use a timing driven compression algorithm.
  // If false, use a simple compression algorithm (e.g., Wallace).
  bool usingTiming;

  // Bitwidth of compressor tree
  const size_t width;

  // Number of reduction stages
  size_t numStages;

  // Number of full adders used
  size_t numFullAdders;

  // Location of compressor to replace
  Location loc;

  SmallVector<Value> columnsToAddends(OpBuilder &builder, size_t targetHeight);

  // Perform timing driven compression using Dadda's algorithm
  SmallVector<Value> compressUsingTiming(OpBuilder &builder,
                                         size_t targetHeight);

  // Perform Wallace tree reduction on partial products.
  // See https://en.wikipedia.org/wiki/Wallace_tree
  SmallVector<Value> compressWithoutTiming(OpBuilder &builder,
                                           size_t targetHeight);

  // Create a full-adder and update delay of sum and carry bits
  std::pair<CompressorBit, CompressorBit> fullAdderWithDelay(OpBuilder &builder,
                                                             CompressorBit a,
                                                             CompressorBit b,
                                                             CompressorBit c);
  // Create a half-adder and update delay of sum and carry bits
  std::pair<CompressorBit, CompressorBit>
  halfAdderWithDelay(OpBuilder &builder, CompressorBit a, CompressorBit b);
};

} // namespace datapath
} // namespace circt

#endif // CIRCT_DIALECT_DATAPATH_DATAPATHOPS_H
