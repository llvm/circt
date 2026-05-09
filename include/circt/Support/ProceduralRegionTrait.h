//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines traits for (non-)procedural regions and operations.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_PROCEDURALREGIONTRAIT_H
#define CIRCT_SUPPORT_PROCEDURALREGIONTRAIT_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/Support/LogicalResult.h"

namespace circt {

/// Returns `success` if the operation has no closer surrounding parent
/// marked as procedural region than its closest parent marked as
/// non-procedural region.
/// Also returns `success` if no parent is marked as either procedural or
/// non-procedural region.
LogicalResult verifyNotInProceduralRegion(Operation *op);

/// Returns `success` if the operation has no closer surrounding parent
/// marked as non-procedural region than its closest parent marked as
/// procedural region.
/// Also returns `success` if no parent is marked as either procedural or
/// non-procedural region.
LogicalResult verifyNotInNonProceduralRegion(Operation *op);

/// Returns true if `op` is itself marked as a procedural region, or has such
/// a parent that is closer than any parent marked as a non-procedural region.
/// Returns false if neither condition holds. Useful for deciding whether ops
/// inserted as children of `op` would live in a procedural region.
bool isProceduralRegionOp(Operation *op);

/// Signals that an operation's regions are procedural.
template <typename ConcreteType>
class ProceduralRegion
    : public mlir::OpTrait::TraitBase<ConcreteType, ProceduralRegion> {
  static LogicalResult verifyTrait(Operation *op) {
    return mlir::OpTrait::impl::verifyNRegions(op, 1);
  }
};

/// Signals that an operation's regions are non-procedural.
template <typename ConcreteType>
class NonProceduralRegion
    : public mlir::OpTrait::TraitBase<ConcreteType, NonProceduralRegion> {
  static LogicalResult verifyTrait(Operation *op) {
    return mlir::OpTrait::impl::verifyNRegions(op, 1);
  }
};

/// Signals that an operation must not be in a non-procedural region.
template <typename ConcreteType>
class ProceduralOp
    : public mlir::OpTrait::TraitBase<ConcreteType, ProceduralOp> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return verifyNotInNonProceduralRegion(op);
  }
};

/// Signals that an operation must not be in a procedural region.
template <typename ConcreteType>
class NonProceduralOp
    : public mlir::OpTrait::TraitBase<ConcreteType, NonProceduralOp> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return verifyNotInProceduralRegion(op);
  }
};

} // namespace circt

#endif // CIRCT_SUPPORT_PROCEDURALREGIONTRAIT_H
