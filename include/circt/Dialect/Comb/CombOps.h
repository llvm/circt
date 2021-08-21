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
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
class PatternRewriter;
}

#define GET_OP_CLASSES
#include "circt/Dialect/Comb/Comb.h.inc"

namespace circt {
namespace comb {
/// KnownBitAnalysis captures information about a value - the set of bits that
/// are guaranteed to always be zero, and the set of bits that are guaranteed to
/// always be one (these must be exclusive!).  A bit that exists in neither
/// set is unknown.
///
/// The main entrypoint to this API is `KnownBitAnalysis::compute(v)`.
///
struct KnownBitAnalysis {
  APInt ones, zeros;

  KnownBitAnalysis(APInt ones, APInt zeros) : ones(ones), zeros(zeros) {}

  static KnownBitAnalysis getUnknown(Value value) {
    auto width = value.getType().getIntOrFloatBitWidth();
    return KnownBitAnalysis{APInt(width, 0), APInt(width, 0)};
  }

  static KnownBitAnalysis getConstant(const APInt &value) {
    return KnownBitAnalysis{value, ~value};
  }

  /// Given an integer SSA value, check to see if we know anything about the
  /// result of the computation.  For example, we know that "and with a
  /// constant" always returns zeros for the zero bits in a constant.
  static KnownBitAnalysis compute(Value v);

  /// Return the bitwidth of the analyzed value.
  unsigned getWidth() const { return ones.getBitWidth(); }

  /// Return true if any bits are known about this value.
  bool areAnyKnown() const { return !(ones | zeros).isNullValue(); }

  /// Return true if all bits are known about this value.
  bool areAllKnown() const { return (ones | zeros).isAllOnesValue(); }

  /// Return the set of all bits that have known values.
  APInt getBitsKnown() const { return ones | zeros; }
};

/// Register Comb passes to print analysis information.
void registerCombAnalysisPasses();

/// Register all Comb-related passes.
void registerCombPasses();

} // namespace comb
} // namespace circt

#endif // CIRCT_DIALECT_COMB_COMBOPS_H
