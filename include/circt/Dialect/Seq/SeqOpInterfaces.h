//===- SeqOpInterfaces.h - Declare Seq op interfaces ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation interfaces for the Seq dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SEQ_SEQOPINTERFACES_H
#define CIRCT_DIALECT_SEQ_SEQOPINTERFACES_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/OpDefinition.h"

#include "circt/Dialect/Seq/SeqOpInterfaces.h.inc"
namespace circt {
namespace seq { 

//===----------------------------------------------------------------------===//
// Other Supporting Logic
//===----------------------------------------------------------------------===//

/// Return true if the specified operation is in a procedural region.
LogicalResult verifyInProceduralRegion(Operation *op);
/// Return true if the specified operation is not in a procedural region.
LogicalResult verifyInNonProceduralRegion(Operation *op);

/// Signals that an operations regions are procedural.
template <typename ConcreteType>
class ProceduralRegion
    : public mlir::OpTrait::TraitBase<ConcreteType, ProceduralRegion> {
  static LogicalResult verifyTrait(Operation *op) {
    return mlir::OpTrait::impl::verifyAtLeastNRegions(op, 1);
  }
};

/// This class verifies that the specified op is located in a procedural region.
template <typename ConcreteType>
class ProceduralOp
    : public mlir::OpTrait::TraitBase<ConcreteType, ProceduralOp> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return verifyInProceduralRegion(op);
  }
};

/// This class verifies that the specified op is not located in a procedural
/// region.
template <typename ConcreteType>
class NonProceduralOp
    : public mlir::OpTrait::TraitBase<ConcreteType, NonProceduralOp> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return verifyInNonProceduralRegion(op);
  }
};

} // namespace seq
} // namespace circt

#endif // CIRCT_DIALECT_SEQ_SEQOPINTERFACES_H
