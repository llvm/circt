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
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace llvm {
struct KnownBits;
}

namespace mlir {
class PatternRewriter;
}

#define GET_OP_CLASSES
#include "circt/Dialect/Comb/Comb.h.inc"

namespace circt {
namespace hw {
class ConstantOp;
} // namespace hw
namespace comb {

using llvm::KnownBits;

/// Compute "known bits" information about the specified value - the set of bits
/// that are guaranteed to always be zero, and the set of bits that are
/// guaranteed to always be one (these must be exclusive!).  A bit that exists
/// in neither set is unknown.
KnownBits computeKnownBits(Value value);

/// Create a sign extension operation from a value of integer type to an equal
/// or larger integer type.
Value createOrFoldSExt(Location loc, Value value, Type destTy,
                       OpBuilder &builder);
Value createOrFoldSExt(Value value, Type destTy, ImplicitLocOpBuilder &builder);

/// Create a ``Not'' gate on a value.
Value createOrFoldNot(Location loc, Value value, OpBuilder &builder);
Value createOrFoldNot(Value value, ImplicitLocOpBuilder &builder);

/// Given a mux `rootMux`, check to see if the "on true" value (or "on false"
/// value if isFalseSide=true) is a mux tree with the same condition.  This
/// allows us to detect a mux tree like `mux(VAL == 0, A, (mux (VAL == 1), B,
/// C))`. Return true if the pattern maching successes, and set results to
/// arguments. For the example above, `indexValue` is `VAL`,  `defaultValue` is
/// C, and `valuesFound` is `{{0, A}, {1, B}}`.
bool getLinearMuxChainsComparison(
    MuxOp rootMux, bool isFalseSide, Value &indexValue, Value &defaultValue,
    SmallVectorImpl<Location> &locationsFound,
    SmallVectorImpl<std::pair<circt::hw::ConstantOp, Value>> &valuesFound);

} // namespace comb
} // namespace circt

#endif // CIRCT_DIALECT_COMB_COMBOPS_H
