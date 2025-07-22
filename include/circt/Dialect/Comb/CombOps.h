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

/// Perform Wallace tree reduction on partial products.
/// See https://en.wikipedia.org/wiki/Wallace_tree
/// \param targetAddends The number of addends to reduce to (2 for carry-save).
/// \param inputAddends The rows of bits to be summed.
SmallVector<Value> wallaceReduction(OpBuilder &builder, Location loc,
                                    size_t width, size_t targetAddends,
                                    SmallVector<SmallVector<Value>> &addends);

} // namespace comb
} // namespace circt

#endif // CIRCT_DIALECT_COMB_COMBOPS_H
