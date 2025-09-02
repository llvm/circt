//===- CombOps.h - Declare Comb dialect operations --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the Comb dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_COMB_COMBOPS_H
#define CIRCT_DIALECT_COMB_COMBOPS_H

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/LLVM.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"

namespace llvm {
struct KnownBits;
}

namespace mlir {
class PatternRewriter;
}

#define GET_OP_CLASSES
#include "circt/Dialect/Comb/Comb.h.inc"

namespace circt {
namespace comb {

using llvm::KnownBits;

/// Compute "known bits" information about the specified value - the set of bits
/// that are guaranteed to always be zero, and the set of bits that are
/// guaranteed to always be one (these must be exclusive!).  A bit that exists
/// in neither set is unknown.
KnownBits computeKnownBits(Value value);

/// Create the ops to zero-extend a value to an integer of equal or larger type.
Value createZExt(OpBuilder &builder, Location loc, Value value,
                 unsigned targetWidth);

/// Create a sign extension operation from a value of integer type to an equal
/// or larger integer type.
Value createOrFoldSExt(Location loc, Value value, Type destTy,
                       OpBuilder &builder);
Value createOrFoldSExt(Value value, Type destTy, ImplicitLocOpBuilder &builder);

/// Create a ``Not'' gate on a value.
Value createOrFoldNot(Location loc, Value value, OpBuilder &builder,
                      bool twoState = false);
Value createOrFoldNot(Value value, ImplicitLocOpBuilder &builder,
                      bool twoState = false);

/// Extract bits from a value.
void extractBits(OpBuilder &builder, Value val, SmallVectorImpl<Value> &bits);

/// Construct a mux tree for given leaf nodes. `selectors` is the selector for
/// each level of the tree. Currently the selector is tested from MSB to LSB.
Value constructMuxTree(OpBuilder &builder, Location loc,
                       ArrayRef<Value> selectors, ArrayRef<Value> leafNodes,
                       Value outOfBoundsValue);

/// Extract a range of bits from an integer at a dynamic offset.
Value createDynamicExtract(OpBuilder &builder, Location loc, Value value,
                           Value offset, unsigned width);

/// Replace a range of bits in an integer at a dynamic offset, and return the
/// updated integer value. Calls `createInject` if the offset is constant.
Value createDynamicInject(OpBuilder &builder, Location loc, Value value,
                          Value offset, Value replacement,
                          bool twoState = false);

/// Replace a range of bits in an integer and return the updated integer value.
Value createInject(OpBuilder &builder, Location loc, Value value,
                   unsigned offset, Value replacement);

/// Construct a full adder for three 1-bit inputs.
std::pair<Value, Value> fullAdder(OpBuilder &builder, Location loc, Value a,
                                  Value b, Value c);
struct CompressorBit {
  Value val;
  size_t delay;
};

std::pair<CompressorBit, CompressorBit>
fullAdderWithDelay(OpBuilder &builder, Location loc, CompressorBit a,
                   CompressorBit b, CompressorBit c);

std::pair<CompressorBit, CompressorBit> halfAdderWithDelay(OpBuilder &builder,
                                                           Location loc,
                                                           CompressorBit a,
                                                           CompressorBit b);
class CompressorTree {
public:
  // Constructor takes addends as input and converts to column representation
  CompressorTree(const SmallVector<SmallVector<Value>> &addends, Location loc);

  // Get the number of columns (bit positions)
  size_t getWidth() const { return columns.size(); }

  // Get the maximum height of the addend array
  size_t getMaxHeight() const;

  // Get the maximum height of the addend array
  void setUsingTiming(bool useTiming) { this->usingTiming = useTiming; }

  // Get the target height of next stage
  size_t getNextStageTargetHeight() const;

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
  // If true, use a timing driven compression algorithm (Dadda's algorithm).
  // If false, use a simple compression algorithm (e.g., Wallace).
  bool usingTiming;

  // Bitwidth of compressor tree
  size_t width;

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

  // Helper method to extract bit at position from a value
  Value extractBit(Value val, unsigned bitPos) const;
};

} // namespace comb
} // namespace circt

#endif // CIRCT_DIALECT_COMB_COMBOPS_H
