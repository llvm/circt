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
namespace comb {

/// This trait declares all regions of the operation it is attached to as
/// non-comb-semantic-perserving. This means, the Comb dialect will not attempt
/// any folding, canonicalization, and optimization across region and block
/// boundaries. More precisely, if a comb operation is defined in block#1 inside
/// region#0 of an operation with this trait, comb canonicalizers will not
/// consider def-use edges coming from outside region#0 as well as outside
/// block#1.
template <typename ConcreteType>
class NonCombSemanticPreservingRegion
    : public mlir::OpTrait::TraitBase<ConcreteType,
                                      NonCombSemanticPreservingRegion> {};

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
Value createOrFoldNot(Location loc, Value value, OpBuilder &builder,
                      bool twoState = false);
Value createOrFoldNot(Value value, ImplicitLocOpBuilder &builder,
                      bool twoState = false);

} // namespace comb
} // namespace circt

#endif // CIRCT_DIALECT_COMB_COMBOPS_H
