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

/// Replace a subtraction with an addition of the two's complement.
LogicalResult convertSubToAdd(comb::SubOp subOp,
                              mlir::PatternRewriter &rewriter);

/// Convert unsigned division or modulo by a power of two.
/// For division: divu(x, 2^n) -> concat(0...0, extract(x, n, width-n)).
/// For modulo: modu(x, 2^n) -> concat(0...0, extract(x, 0, n))
/// TODO: Support signed division and modulo.
LogicalResult convertDivUByPowerOfTwo(DivUOp divOp,
                                      mlir::PatternRewriter &rewriter);
LogicalResult convertModUByPowerOfTwo(ModUOp modOp,
                                      mlir::PatternRewriter &rewriter);

/// Enum for mux chain folding styles.
enum MuxChainWithComparisonFoldingStyle { None, BalancedMuxTree, ArrayGet };
/// Mux chain folding that converts chains of muxes with index
/// comparisons into array operations or balanced mux trees. `styleFn` is a
/// callback that returns the desired folding style based on the index
/// width and number of entries.
bool foldMuxChainWithComparison(
    PatternRewriter &rewriter, MuxOp rootMux, bool isFalseSide,
    llvm::function_ref<MuxChainWithComparisonFoldingStyle(size_t indexWidth,
                                                          size_t numEntries)>
        styleFn);

// Check if the operand is sext() and return the unextended operand:
// signBit = comb.extract(baseValue, width-1, 1)
// ext = comb.replicate(signBit, width-baseWidth)
// sext = comb.concat(ext, baseValue)
template <typename SubType>
struct SextMatcher {
  SubType lhs;
  SextMatcher(SubType lhs) : lhs(std::move(lhs)) {}
  bool match(Operation *op) {
    // Check if operand is a concat operation
    auto concatOp = dyn_cast<ConcatOp>(op);
    if (!concatOp)
      return false;

    auto operands = concatOp.getOperands();
    // ConcatOp must have exactly 2 operands: (sign_bits, base_value)
    if (operands.size() != 2)
      return false;

    Value signBits = operands[0];
    Value baseValue = operands[1];
    auto baseWidth = baseValue.getType().getIntOrFloatBitWidth();

    // Check if signBits is a replicate operation
    auto replicateOp = dyn_cast<ReplicateOp>(signBits.getDefiningOp());
    if (!replicateOp)
      return false;

    Value signBit = replicateOp.getInput();

    // Check if signBit is the msb of baseValue
    auto extractOp = dyn_cast<ExtractOp>(signBit.getDefiningOp());
    if (!extractOp)
      return false;

    if ((extractOp.getInput() != baseValue) ||
        (extractOp.getLowBit() != baseWidth - 1) ||
        (extractOp.getType().getIntOrFloatBitWidth() != 1))
      return false;

    // Match the base unextended value against the sub-matcher
    return lhs.match(op->getOperand(1));
  }
};

/// Helper function to create a sign extension matcher
template <typename SubType>
static inline SextMatcher<SubType> m_Sext(const SubType &subExpr) {
  return SextMatcher<SubType>(subExpr);
}

} // namespace comb
} // namespace circt

#endif // CIRCT_DIALECT_COMB_COMBOPS_H
