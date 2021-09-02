//===- BitwidthAnalysis.cpp - Support for building backedges ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides flow-based forward bitwidth analysis.
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/BitwidthAnalysis.h"
#include "mlir/Analysis/DataFlowAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"

static unsigned s_saturationWidth = 64;

using namespace mlir;

namespace circt {
namespace bitwidth {
// ============================================================================
// Utility functions
// ============================================================================

/// Returns the minimum of two APInt's. In case of mismatched bit width, return
/// the value with the largest bit width.
static APInt min(const APInt &lhs, const APInt &rhs) {
  unsigned lbits = lhs.getBitWidth();
  unsigned rbits = rhs.getBitWidth();
  if (lbits != rbits)
    return lbits < rbits ? rhs : lhs;
  else
    return lhs.slt(rhs) ? lhs : rhs;
}

/// Returns the maximum of two APInt's. In case of mismatched bit width, return
/// the value with the largest bit width.
static APInt max(const APInt &lhs, const APInt &rhs) {
  unsigned lbits = lhs.getBitWidth();
  unsigned rbits = rhs.getBitWidth();
  if (lbits != rbits)
    return lbits < rbits ? rhs : lhs;
  else
    return lhs == min(lhs, rhs) ? rhs : lhs;
}

/// Returns the number of bits needed to represent the value @p v.
static unsigned bits(const APInt &v) {
  unsigned n = v.logBase2() + 1;
  return n == 0 ? 1 : n;
}

/// The ForwardDataFlowAnalysis solver does not consider loops in the control
/// flow graph and will pre-emptively converge on values within a loop. This
/// helper function performs a DFS to analyse if a block resides within a loop.
static bool inCycle(Block *srcBlock) {
  llvm::SmallSet<Block *, 8> visited;
  std::function<bool(Block *)> cycleUtil = [&](Block *block) {
    if (block == srcBlock)
      return true;
    if (visited.contains(block))
      return false;
    visited.insert(block);
    return llvm::any_of(block->getSuccessors(), cycleUtil);
  };
  return llvm::any_of(srcBlock->getSuccessors(), cycleUtil);
}

/// Returns true if any value in @p operands is defined within a control-flow
/// cycle.
static bool anyOpInCycle(ValueRange operands) {
  return llvm::any_of(operands, [](Value op) {
    if (auto barg = op.dyn_cast<BlockArgument>())
      return inCycle(barg.getOwner());
    else
      return inCycle(op.getDefiningOp()->getBlock());
  });
}

/// Attempts to match a constant op and the value which it defines. If
/// successfull, binds the constant value to @p value and returns true, else,
/// returns false.
static bool matchConstantOp(Operation *op, APInt &value) {
  return mlir::detail::constant_int_op_binder(&value).match(op);
}

} // namespace bitwidth

// ============================================================================
// BitWidthLattice
// ============================================================================
/// Lattice value for the dataflow algorithm.
struct BitWidthLattice {
  BitWidthLattice() {}
  BitWidthLattice(APInt lmin, APInt lmax) : lmin(lmin), lmax(lmax) {}

  bool operator==(const BitWidthLattice &rhs) const {
    return (this->lmin.getBitWidth() == rhs.lmin.getBitWidth() &&
            this->lmin == rhs.lmin) &&
           (this->lmax.getBitWidth() == rhs.lmax.getBitWidth() &&
            this->lmax == rhs.lmax);
  }

  /// Perorms a range-join of the lhs and rhs minimum and maximum values.
  static BitWidthLattice join(const BitWidthLattice &lhs,
                              const BitWidthLattice &rhs) {
    return BitWidthLattice(bitwidth::min(lhs.lmin, rhs.lmin),
                           bitwidth::max(lhs.lmax, rhs.lmax));
  }

  /// The (most) pessimistic value is our saturation width.
  static BitWidthLattice getPessimisticValueState(mlir::MLIRContext *);

  /// Try to infer a bit width from the type of a value, else, fallback to
  /// saturation width.
  static BitWidthLattice getPessimisticValueState(mlir::Value value);

  /// Return the number of bits required to represent the value range of this
  /// lattice.
  unsigned bits() const {
    return std::max(bitwidth::bits(lmin), bitwidth::bits(lmax));
  }

  /// Minimum (lower) and maximum (upper) bounds which determines the value
  /// range of this lattice. Values are inclusive and signed.
  APInt lmin, lmax;
};

/// Returns a signed integer lattice with minimum and maximum values determined
/// by @p nbits.
static BitWidthLattice nbitLattice(unsigned nbits) {
  return BitWidthLattice(APInt::getSignedMinValue(nbits),
                         APInt::getSignedMaxValue(nbits));
}

/// Returns a signed integer lattice with minimum and maximum values determined
/// by the width of the integer type @p t.
static BitWidthLattice latticeForIntType(Type t) {
  assert(t.isa<IntegerType>());
  BitWidthLattice lattice;
  unsigned nbits = t.getIntOrFloatBitWidth();
  return nbitLattice(nbits);
}

BitWidthLattice BitWidthLattice::getPessimisticValueState(mlir::MLIRContext *) {
  return nbitLattice(s_saturationWidth);
}

BitWidthLattice BitWidthLattice::getPessimisticValueState(mlir::Value value) {
  auto type = value.getType();
  unsigned nbits = s_saturationWidth;
  if (type.isa<IntegerType>()) {
    nbits = type.getIntOrFloatBitWidth();
    return BitWidthLattice(APInt::getSignedMinValue(nbits),
                           APInt::getSignedMaxValue(nbits));
  }
  if (type.isa<FloatType>())
    nbits = type.getIntOrFloatBitWidth();

  return nbitLattice(nbits);
}

// ============================================================================
// BitWidthAnalysis and visitors
// ============================================================================

/// The AnalyzedOperand struct wraps a value (typically an operand of an
/// operation) and its corresponding (at the time of instantiation)
/// BitWidthLattice to provide various operations for determining the bounds of
/// the value.
struct AnalyzedOperand {
public:
  Value v;
  const BitWidthLattice &lattice;

  /// Returns the number of bits that is currently estimated to be required to
  /// represent @v.
  unsigned bits() const { return lattice.bits(); }

  /// Returns the currently estimated signed maximum value of @v.
  int64_t smax() const {
    if (auto constVal = cval()) {
      return constVal.getValue().getSExtValue();
    }
    return APInt::getSignedMaxValue(lattice.lmax.getBitWidth()).getSExtValue();
  }

  /// Returns the currently estimated signed minimum value of @v.
  int64_t smin() const {
    if (auto constVal = cval()) {
      return constVal.getValue().getSExtValue();
    }
    return APInt::getSignedMinValue(lattice.lmin.getBitWidth()).getSExtValue();
  }

  /// Returns the currently estimated unsigned maximum value of @v.
  uint64_t umax() const {
    if (auto constVal = cval()) {
      return constVal.getValue().getZExtValue();
    }
    return APInt::getMaxValue(lattice.lmax.getBitWidth()).getZExtValue();
  }

  /// Returns the currently estimated unsigned minimum value of @v.
  uint64_t umin() const {
    if (auto constVal = cval()) {
      return constVal.getValue().getZExtValue();
    }
    return APInt::getMinValue(lattice.lmin.getBitWidth()).getZExtValue();
  }

  bool isConstant() const { return cval().hasValue(); }

private:
  /// Returns the constant value which @v represents, if any
  Optional<APInt> cval() const {
    if (auto *defOp = v.getDefiningOp()) {
      APInt constVal;
      if (bitwidth::matchConstantOp(defOp, constVal))
        return constVal;
    }
    return {};
  }
};

/// Bit width rules for various binary operators
/// @todo: as a future improvements, operators could implement an interface for
/// describing their bit-width modifying characteristics.
template <typename TBinOp>
unsigned visitBinOp(const AnalyzedOperand &lhs, const AnalyzedOperand &rhs);

template <>
unsigned visitBinOp<AddIOp>(const AnalyzedOperand &lhs,
                            const AnalyzedOperand &rhs) {
  return std::max(lhs.bits(), rhs.bits()) + 1;
}
template <>
unsigned visitBinOp<SubIOp>(const AnalyzedOperand &lhs,
                            const AnalyzedOperand &rhs) {
  return std::max(lhs.bits(), rhs.bits()) + 1;
}
template <>
unsigned visitBinOp<MulIOp>(const AnalyzedOperand &lhs,
                            const AnalyzedOperand &rhs) {
  return lhs.bits() + rhs.bits();
}
template <>
unsigned visitBinOp<ShiftLeftOp>(const AnalyzedOperand &lhs,
                                 const AnalyzedOperand &rhs) {
  // define a cut-off point for width extension as 2*saturation width. This is
  // relevant when shifting by a non-constant, wide rhs value. In
  // these cases, default to the lhs bit width.
  unsigned lhsBits = lhs.bits();
  uint64_t resBits = lhsBits + rhs.umax();
  if (!rhs.isConstant() && resBits > 2 * s_saturationWidth)
    return lhsBits;
  return resBits;
}
unsigned visitRightShift(const AnalyzedOperand &lhs,
                         const AnalyzedOperand &rhs) {
  int64_t rhsMin = rhs.umin();
  int64_t resBits = lhs.bits() - rhsMin;
  if (rhsMin >= resBits)
    return 1;
  return resBits;
}
template <>
unsigned visitBinOp<SignedShiftRightOp>(const AnalyzedOperand &lhs,
                                        const AnalyzedOperand &rhs) {
  return visitRightShift(lhs, rhs);
}
template <>
unsigned visitBinOp<UnsignedShiftRightOp>(const AnalyzedOperand &lhs,
                                          const AnalyzedOperand &rhs) {
  return visitRightShift(lhs, rhs);
}

class BitWidthAnalysisImpl
    : public mlir::ForwardDataFlowAnalysis<BitWidthLattice> {
public:
  explicit BitWidthAnalysisImpl(mlir::MLIRContext *context)
      : ForwardDataFlowAnalysis(context) {}

  mlir::ChangeResult visitOperation(
      Operation *op,
      ArrayRef<LatticeElement<BitWidthLattice> *> operands) override {
    APInt value;

    /// Since this is a forward dataflow pass, values inside loops will
    /// pre-emptively be marked as converged to a possibly incorrect value. Any
    /// operation which has an operand that is defined within a loop is
    /// conservatively visited by the default handler.
    if (bitwidth::anyOpInCycle(op->getOperands()))
      return visitDefault(op);

    /// If we can statically determine that an operation defines a constant, we
    /// define strict value constraints through visitConstant
    if (bitwidth::matchConstantOp(op, value))
      return visitConstant(op, value);

    /// Try execute operation-specific visitors, or fallback to the default
    /// visitor.
    return TypeSwitch<Operation *, ChangeResult>(op)
        .Case<AddIOp, MulIOp, SubIOp, ShiftLeftOp, SignedShiftRightOp,
              UnsignedShiftRightOp>(
            [&](auto op_t) { return binOpVisitorWrapper(op_t, operands); })
        .Default([&](auto op_t) { return visitDefault(op_t); });
  }

private:
  /// Default handling for determining the constraints on the results of an
  /// operation, if no other visitor is able to provide operation-specific
  /// bitwidth inference logic.
  ChangeResult visitDefault(Operation *op);

  /// visitConstant will set strict upper- lower- and bit width constraints
  /// based on the constant value.
  ChangeResult visitConstant(Operation *op, APInt constVal);

  /// Generic wrapper for binary operators. Unpacks the BitWidthLattice's from
  /// the operands, constructs a new nbitLattice from the bit result of the
  /// specialized operand function, and joins this with the current lattice of
  /// the operand result.
  template <typename OpType>
  ChangeResult
  binOpVisitorWrapper(OpType op,
                      ArrayRef<LatticeElement<BitWidthLattice> *> operands) {
    BitWidthLattice lattice;
    assert(operands.size() == 2);
    auto op0 = AnalyzedOperand{op.getOperand(0), operands[0]->getValue()};
    auto op1 = AnalyzedOperand{op.getOperand(1), operands[1]->getValue()};
    unsigned nbits = visitBinOp<OpType>(op0, op1);
    return getLatticeElement(op.getResult()).join(nbitLattice(nbits));
  }
};

ChangeResult BitWidthAnalysisImpl::visitDefault(Operation *op) {
  ChangeResult chResult = ChangeResult::NoChange;

  // Fallback to the pessimistic fixpoint, or type-defined width (if applicable)
  for (auto opRes : op->getResults()) {
    auto type = opRes.getType();
    if (type.isa<IntegerType>())
      chResult |= getLatticeElement(opRes).join(latticeForIntType(type));
    else
      chResult |= getLatticeElement(opRes).markPessimisticFixpoint();
  }
  return chResult;
}

ChangeResult BitWidthAnalysisImpl::visitConstant(Operation *op,
                                                 APInt constVal) {
  assert(op->getResults().size() == 1);
  unsigned nbits = bitwidth::bits(constVal);
  return getLatticeElement(op->getResult(0)).join(nbitLattice(nbits));
}

// ============================================================================
// BitwidthAnalysis driver
// ============================================================================

Optional<unsigned> BitwidthAnalysis::valueWidth(Value v) const {
  if (auto elem = analysis->lookupLatticeElement(v))
    return elem->getValue().bits();
  return {};
}

BitwidthAnalysis::BitwidthAnalysis(Operation *op, unsigned saturationWidth) {
  // Set the global saturation width (required within the static
  // getPessimisticValueState functions).
  s_saturationWidth = saturationWidth;
  analysis = std::make_shared<BitWidthAnalysisImpl>(op->getContext());
  analysis->run(op);
}

} // namespace circt
