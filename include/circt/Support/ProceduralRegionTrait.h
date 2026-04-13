//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines the `ProceduralRegion` trait.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_PROCEDURALREGIONTRAIT_H
#define CIRCT_SUPPORT_PROCEDURALREGIONTRAIT_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/Support/LogicalResult.h"

namespace circt {

LogicalResult verifyNotInProceduralRegion(Operation *op);
LogicalResult verifyNotInNonProceduralRegion(Operation *op);

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
