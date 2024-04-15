//===- HWOpInterfaces.h - Declare HW op interfaces --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation interfaces for the HW dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_HWOPINTERFACES_H
#define CIRCT_DIALECT_HW_HWOPINTERFACES_H

#include "circt/Dialect/HW/HWInstanceImplementation.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/HW/InnerSymbolTable.h"
#include "circt/Support/InstanceGraphInterface.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"

namespace circt {
namespace hw {

void populateHWModuleLikeTypeConversionPattern(StringRef moduleLikeOpName,
                                               RewritePatternSet &patterns,
                                               TypeConverter &converter);

class InnerSymbolOpInterface;
/// Verification hook for verifying InnerSym Attribute.
LogicalResult verifyInnerSymAttr(InnerSymbolOpInterface op);

namespace detail {
LogicalResult verifyInnerRefNamespace(Operation *op);
} // namespace detail

/// Classify operations that are InnerRefNamespace-like,
/// until structure is in place to do this via Traits.
/// Useful for getParentOfType<>, or scheduling passes.
/// Prefer putting the trait on operations here or downstream.
struct InnerRefNamespaceLike {
  /// Return if this operation is explicitly an IRN or appears compatible.
  static bool classof(mlir::Operation *op);
  /// Return if this operation is explicitly an IRN or appears compatible.
  static bool classof(const mlir::RegisteredOperationName *opInfo);
};

/// Classify operations that are InnerSymbolTable-like,
/// until structure is in place to do this via Traits.
/// Useful for getParentOfType<>, or scheduling passes.
/// Prefer putting the trait on operations here or downstream.
struct InnerSymbolTableLike {
  /// Return if this operation is explicitly an IRN or appears compatible.
  static bool classof(mlir::Operation *op);
  /// Return if this operation is explicitly an IRN or appears compatible.
  static bool classof(const mlir::RegisteredOperationName *opInfo);
};

} // namespace hw
} // namespace circt

namespace mlir {
namespace OpTrait {

/// This trait is for operations that define a scope for resolving InnerRef's,
/// and provides verification for InnerRef users (via InnerRefUserOpInterface).
template <typename ConcreteType>
class InnerRefNamespace : public TraitBase<ConcreteType, InnerRefNamespace> {
public:
  static LogicalResult verifyRegionTrait(Operation *op) {
    if (op->getNumRegions() != 1)
      return op->emitError("expected operation to have a single region");
    if (!op->getRegion(0).hasOneBlock())
      return op->emitError("expected operation to have a single block");

    // Verify all InnerSymbolTable's and InnerRef users.
    return ::circt::hw::detail::verifyInnerRefNamespace(op);
  }
};

/// A trait for inner symbol table functionality on an operation.
template <typename ConcreteType>
class InnerSymbolTable : public TraitBase<ConcreteType, InnerSymbolTable> {
public:
  static LogicalResult verifyRegionTrait(Operation *op);
};
} // namespace OpTrait
} // namespace mlir

#include "circt/Dialect/HW/HWOpInterfaces.h.inc"

namespace mlir {
namespace OpTrait {
template <typename ConcreteType>
LogicalResult InnerSymbolTable<ConcreteType>::verifyRegionTrait(Operation *op) {
  // Insist that ops with InnerSymbolTable's provide either a symbol or inner
  // symbol, this is essential to how InnerRef's work.
  static_assert(
      ConcreteType::template hasTrait<
          circt::hw::InnerSymbolOpInterface::Trait>() ||
          ConcreteType::template hasTrait<mlir::SymbolOpInterface::Trait>(),
      "expected operation to define either an InnerSymbol or Symbol");

  // InnerSymbolTable's must be directly nested under either another
  // InnerSymbolTable operation or an InnerRefNamespace.
  auto *parent = op->getParentOp();
  if (!parent ||
      !isa<circt::hw::InnerRefNamespaceLike, circt::hw::InnerSymbolTableLike>(
          parent)) {
    return op->emitError("InnerSymbolTable must have InnerRefNamespace or "
                         "InnerSymbolTable as parent");
  }

  return success();
}
} // namespace OpTrait
} // namespace mlir

#endif // CIRCT_DIALECT_HW_HWOPINTERFACES_H
