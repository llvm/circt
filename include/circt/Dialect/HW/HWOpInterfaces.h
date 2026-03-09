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

#include "circt/Dialect/Emit/EmitOpInterfaces.h"
#include "circt/Dialect/HW/HWInstanceImplementation.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/HW/InnerSymbolTable.h"
#include "circt/Support/FieldRef.h"
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
/// Verification hook for verifying InnerSymbol-defining operations.
LogicalResult verifyInnerSymOp(InnerSymbolOpInterface op);

/// Verification hook for verifying InnerSymbol-defining operations.
LogicalResult verifyPortInnerSymsIfPortList(Operation *op);

// Verification hook for verifying an inner symbol attribute.
// Variant accepting 'Type' is for per-field checking.
// (for when the inner symbol points into fields of a type)
LogicalResult
verifyInnerSymAttr(InnerSymAttr innerSym, Type type,
                   llvm::function_ref<InFlightDiagnostic()> emitError);
inline LogicalResult
verifyInnerSymAttr(InnerSymAttr innerSym,
                   llvm::function_ref<InFlightDiagnostic()> emitError) {
  return verifyInnerSymAttr(innerSym, {}, emitError);
}

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
    static_assert(
        ConcreteType::template hasTrait<::mlir::OpTrait::SymbolTable>(),
        "expected operation to be a SymbolTable");

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
  static LogicalResult verifyRegionTrait(Operation *op) {
    // Insist that ops with InnerSymbolTable's provide a Symbol, this is
    // essential to how InnerRef's work.
    static_assert(
        ConcreteType::template hasTrait<::mlir::SymbolOpInterface::Trait>(),
        "expected operation to define a Symbol");

    // InnerSymbolTable's must be directly nested within an InnerRefNamespace.
    auto *parent = op->getParentOp();
    if (!parent || !isa<circt::hw::InnerRefNamespaceLike>(parent))
      return op->emitError(
          "InnerSymbolTable must have InnerRefNamespace parent");

    // If also a PortList, walk and verify each port's inner symbol(s).
    return ::circt::hw::verifyPortInnerSymsIfPortList(op);
  }
};
} // namespace OpTrait
} // namespace mlir

#include "circt/Dialect/HW/HWOpInterfaces.h.inc"

#endif // CIRCT_DIALECT_HW_HWOPINTERFACES_H
