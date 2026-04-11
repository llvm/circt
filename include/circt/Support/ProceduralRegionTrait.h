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

#include "mlir/IR/OpDefinition.h"
#include "llvm/Support/LogicalResult.h"

namespace circt {

/// Signals that an operation's regions are procedural.
template <typename ConcreteType>
class ProceduralRegion
    : public mlir::OpTrait::TraitBase<ConcreteType, ProceduralRegion> {
  static LogicalResult verifyTrait(Operation *op) {
    return mlir::OpTrait::impl::verifyNRegions(op, 1);
  }
};

} // namespace circt

#endif // CIRCT_SUPPORT_PROCEDURALREGIONTRAIT_H
