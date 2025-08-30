//===- CombAnalysis.cpp - Analysis Helpers for Comb+HW operations ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "llvm/Support/KnownBits.h"

using namespace circt;
using namespace comb;

/// Given an integer SSA value, check to see if we know anything about the
/// result of the computation.  For example, we know that "and with a constant"
/// always returns zeros for the zero bits in a constant.
///
/// Expression trees can be very large, so we need ot make sure to cap our
/// recursion, this is controlled by `depth`.
static KnownBits computeKnownBits(Value v, unsigned depth) {
  Operation *op = v.getDefiningOp();
  if (!op || depth == 5)
    return KnownBits(v.getType().getIntOrFloatBitWidth());

  // A constant has all bits known!
  if (auto constant = dyn_cast<hw::ConstantOp>(op))
    return KnownBits::makeConstant(constant.getValue());

  // `concat(x, y, z)` has whatever is known about the operands concat'd.
  if (auto concatOp = dyn_cast<ConcatOp>(op)) {
    auto result = computeKnownBits(concatOp.getOperand(0), depth + 1);
    for (size_t i = 1, e = concatOp.getNumOperands(); i != e; ++i) {
      auto otherBits = computeKnownBits(concatOp.getOperand(i), depth + 1);
      unsigned width = otherBits.getBitWidth();
      unsigned newWidth = result.getBitWidth() + width;
      result.Zero =
          (result.Zero.zext(newWidth) << width) | otherBits.Zero.zext(newWidth);
      result.One =
          (result.One.zext(newWidth) << width) | otherBits.One.zext(newWidth);
    }
    return result;
  }

  // `extract(x from y)` has whatever is known about the extracted bits of the
  // operand.
  if (auto extractOp = dyn_cast<ExtractOp>(op)) {
    unsigned lowBit = extractOp.getLowBit();
    unsigned width = extractOp.getType().getIntOrFloatBitWidth();
    // Compute known bits of the input
    KnownBits inputBits = computeKnownBits(extractOp.getInput(), depth + 1);
    // Shift down to start at the extract position
    APInt inputZero = inputBits.Zero.lshr(lowBit);
    APInt inputOne = inputBits.One.lshr(lowBit);

    // Truncate to the width of the extract result
    APInt knownZero = inputZero.trunc(width);
    APInt knownOne = inputOne.trunc(width);

    KnownBits result(width);
    result.Zero = knownZero;
    result.One = knownOne;
    return result;
  }

  // `and(x, y, z)` has whatever is known about the operands intersected.
  if (auto andOp = dyn_cast<AndOp>(op)) {
    auto result = computeKnownBits(andOp.getOperand(0), depth + 1);
    for (size_t i = 1, e = andOp.getNumOperands(); i != e; ++i)
      result &= computeKnownBits(andOp.getOperand(i), depth + 1);
    return result;
  }

  // `or(x, y, z)` has whatever is known about the operands unioned.
  if (auto orOp = dyn_cast<OrOp>(op)) {
    auto result = computeKnownBits(orOp.getOperand(0), depth + 1);
    for (size_t i = 1, e = orOp.getNumOperands(); i != e; ++i)
      result |= computeKnownBits(orOp.getOperand(i), depth + 1);
    return result;
  }

  // `xor(x, cst)` inverts known bits and passes through unmodified ones.
  if (auto xorOp = dyn_cast<XorOp>(op)) {
    auto result = computeKnownBits(xorOp.getOperand(0), depth + 1);
    for (size_t i = 1, e = xorOp.getNumOperands(); i != e; ++i) {
      // If we don't know anything, we don't need to evaluate more subexprs.
      if (result.isUnknown())
        return result;
      result ^= computeKnownBits(xorOp.getOperand(i), depth + 1);
    }
    return result;
  }

  // `mux(cond, x, y)` is the intersection of the known bits of `x` and `y`.
  if (auto muxOp = dyn_cast<MuxOp>(op)) {
    auto lhs = computeKnownBits(muxOp.getTrueValue(), depth + 1);
    auto rhs = computeKnownBits(muxOp.getFalseValue(), depth + 1);
    return lhs.intersectWith(rhs);
  }

  // `shl(x, y)` is the known bits of `x` << known bits of `y`.
  if (auto shlOp = dyn_cast<ShlOp>(op)) {
    auto lhs = computeKnownBits(shlOp.getOperand(0), depth + 1);
    auto rhs = computeKnownBits(shlOp.getOperand(1), depth + 1);
    auto res = KnownBits::shl(lhs, rhs);
    return res;
  }

  // `shr(x, y)` is the known bits of `x` >> known bits of `y`.
  if (auto shrOp = dyn_cast<ShrUOp>(op)) {
    auto lhs = computeKnownBits(shrOp.getOperand(0), depth + 1);
    auto rhs = computeKnownBits(shrOp.getOperand(1), depth + 1);
    auto res = KnownBits::lshr(lhs, rhs);
    return res;
  }

  return KnownBits(v.getType().getIntOrFloatBitWidth());
}

/// Given an integer SSA value, check to see if we know anything about the
/// result of the computation.  For example, we know that "and with a
/// constant" always returns zeros for the zero bits in a constant.
KnownBits comb::computeKnownBits(Value value) {
  return ::computeKnownBits(value, 0);
}
