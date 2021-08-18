//===- CombAnalysis.cpp - Analysis Helpers for Comb+HW operations ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"

using namespace circt;
using namespace comb;

/// Given an integer SSA value, check to see if we know anything about the
/// result of the computation.  For example, we know that "and with a constant"
/// always returns zeros for the zero bits in a constant.
///
/// Expression trees can be very large, so we need ot make sure to cap our
/// recursion, this is controlled by `depth`.
static KnownBitAnalysis computeKnownBits(Value v, unsigned depth) {
  Operation *op = v.getDefiningOp();
  if (!op || depth == 5)
    return KnownBitAnalysis::getUnknown(v);

  // A constant has all bits known!
  if (auto constant = dyn_cast<hw::ConstantOp>(op))
    return KnownBitAnalysis::getConstant(constant.getValue());

  // `concat(x, y, z)` has whatever is known about the operands concat'd.
  if (auto concatOp = dyn_cast<ConcatOp>(op)) {
    auto result = computeKnownBits(concatOp.getOperand(0), depth + 1);
    for (size_t i = 1, e = concatOp.getNumOperands(); i != e; ++i) {
      auto otherBits = computeKnownBits(concatOp.getOperand(i), depth + 1);
      unsigned width = otherBits.getWidth();
      unsigned newWidth = result.getWidth() + width;
      result.zeros = (result.zeros.zext(newWidth) << width) |
                     otherBits.zeros.zext(newWidth);
      result.ones =
          (result.ones.zext(newWidth) << width) | otherBits.ones.zext(newWidth);
    }
    return result;
  }

  // `and(x, y, z)` has whatever is known about the operands intersected.
  if (auto andOp = dyn_cast<AndOp>(op)) {
    auto result = computeKnownBits(andOp.getOperand(0), depth + 1);
    for (size_t i = 1, e = andOp.getNumOperands(); i != e; ++i) {
      auto otherBits = computeKnownBits(andOp.getOperand(i), depth + 1);
      result.zeros |= otherBits.zeros;
      result.ones &= otherBits.ones;
    }
    return result;
  }

  // `or(x, y, z)` has whatever is known about the operands unioned.
  if (auto orOp = dyn_cast<OrOp>(op)) {
    auto result = computeKnownBits(orOp.getOperand(0), depth + 1);
    for (size_t i = 1, e = orOp.getNumOperands(); i != e; ++i) {
      auto otherBits = computeKnownBits(orOp.getOperand(i), depth + 1);
      result.zeros &= otherBits.zeros;
      result.ones |= otherBits.ones;
    }
    return result;
  }

  // TODO: Handle xor(x, cst) by inverting known bits.

  return KnownBitAnalysis::getUnknown(v);
}

/// Given an integer SSA value, check to see if we know anything about the
/// result of the computation.  For example, we know that "and with a
/// constant" always returns zeros for the zero bits in a constant.
KnownBitAnalysis KnownBitAnalysis::compute(Value v) {
  return computeKnownBits(v, 0);
}