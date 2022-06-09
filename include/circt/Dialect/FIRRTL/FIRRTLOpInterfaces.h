//===- FIRRTLOpInterfaces.h - Declare FIRRTL op interfaces ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation interfaces for the FIRRTL IR and supporting
// types.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_OP_INTERFACES_H
#define CIRCT_DIALECT_FIRRTL_OP_INTERFACES_H

#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/Support/CommandLine.h"

namespace circt {
namespace firrtl {

class FIRRTLType;

/// This holds the name and type that describes the module's ports.
struct PortInfo {
  StringAttr name;
  FIRRTLType type;
  Direction direction;
  StringAttr sym = {};
  Location loc = UnknownLoc::get(type.getContext());
  AnnotationSet annotations = AnnotationSet(type.getContext());

  StringRef getName() const { return name ? name.getValue() : ""; }

  /// Return true if this is a simple output-only port.  If you want the
  /// direction of the port, use the \p direction parameter.
  bool isOutput() {
    auto flags = type.getRecursiveTypeProperties();
    return flags.isPassive && !flags.containsAnalog &&
           direction == Direction::Out;
  }

  /// Return true if this is a simple input-only port.  If you want the
  /// direction of the port, use the \p direction parameter.
  bool isInput() {
    auto flags = type.getRecursiveTypeProperties();
    return flags.isPassive && !flags.containsAnalog &&
           direction == Direction::In;
  }

  /// Return true if this is an inout port.  This will be true if the port
  /// contains either bi-directional signals or analog types.
  bool isInOut() { return !isOutput() && !isInput(); }
};

/// Verification hook for verifying module like operations.
LogicalResult verifyModuleLikeOpInterface(FModuleLike module);

namespace detail {
LogicalResult verifyInnerRefs(Operation *op);
} // namespace detail

/// A table of inner symbols and their resolutions.
class InnerSymbolTable {
public:
  /// Return the name of the attribute used for inner symbol names.
  static StringRef getInnerSymbolAttrName() { return "inner_sym"; }

  /// Build an inner symbol table for the given operation.  The operation must
  /// have the InnerSymbolTable trait.
  explicit InnerSymbolTable(Operation *op);

  /// Look up a symbol with the specified name, returning null if no such
  /// name exists. Names never include the @ on them.
  Operation *lookup(StringRef name) const;
  template <typename T>
  T lookup(StringRef name) const {
    return dyn_cast_or_null<T>(lookup(name));
  }

  /// Look up a symbol with the specified name, returning null if no such
  /// name exists. Names never include the @ on them.
  Operation *lookup(StringAttr name) const;
  template <typename T>
  T lookup(StringAttr name) const {
    return dyn_cast_or_null<T>(lookup(name));
  }

private:
  /// This is the operation this table is constructed for, which must have the
  /// InnerSymbolTable trait.
  Operation *innerSymTblOp;

  /// This maps names to operations with that inner symbol.
  DenseMap<StringAttr, Operation *> symbolTable;
};

/// This class represents a collection of InnerSymbolTable's.
class InnerSymbolTableCollection {
public:
  /// Get or create the InnerSymbolTable for the specified operation.
  InnerSymbolTable &getInnerSymbolTable(Operation *op);

  /// Populate tables in parallel for all InnerSymbolTable operations in the
  /// given InnerRefNamespace operation.
  void populateTables(Operation *innerRefNSOp);

private:
  /// This maps Operations to their InnnerSymbolTable's.
  DenseMap<Operation *, std::unique_ptr<InnerSymbolTable>> symbolTables;
};

/// This class represents the namespace in which InnerRef's can be resolved.
struct InnerRefNamespace {
  SymbolTable &symTable;
  InnerSymbolTableCollection &innerSymTables;

  /// Resolve the InnerRef to its target within this namespace, returning null
  /// if no such name exists.
  ///
  /// Note that some InnerRef's target ports and must be handled separately.
  Operation *lookup(hw::InnerRefAttr inner);
  template <typename T>
  T lookup(hw::InnerRefAttr inner) {
    return dyn_cast_or_null<T>(lookup(inner));
  }
};

} // namespace firrtl
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

    // Verify all InnerRef users.
    return ::circt::firrtl::detail::verifyInnerRefs(op);
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

    if (op->getNumRegions() != 1)
      return op->emitError("expected operation to have a single region");
    if (!op->getRegion(0).hasOneBlock())
      return op->emitError("expected operation to have a single block");

    // InnerSymbolTable's must be directly nested within an InnerRefNamespace.
    auto *parent = op->getParentOp();
    if (!parent || !parent->hasTrait<InnerRefNamespace>())
      return op->emitError(
          "InnerSymbolTable must have InnerRefNamespace parent");

    return success();
  }
};
} // namespace OpTrait
} // namespace mlir

#include "circt/Dialect/FIRRTL/FIRRTLOpInterfaces.h.inc"
#endif // CIRCT_DIALECT_FIRRTL_OP_INTERFACES_H
