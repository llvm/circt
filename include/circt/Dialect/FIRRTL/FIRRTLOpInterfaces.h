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
#include "circt/Dialect/FIRRTL/InnerSymbolTable.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"

namespace circt {
namespace firrtl {

class FIRRTLType;

/// This holds the name and type that describes the module's ports.
struct PortInfo {
  StringAttr name;
  Type type;
  Direction direction;
  hw::InnerSymAttr sym = {};
  Location loc = UnknownLoc::get(type.getContext());
  AnnotationSet annotations = AnnotationSet(type.getContext());

  StringRef getName() const { return name ? name.getValue() : ""; }

  /// Return true if this is a simple output-only port.  If you want the
  /// direction of the port, use the \p direction parameter.
  bool isOutput() {
    if (direction != Direction::Out)
      return false;
    if (type.isa<FIRRTLType>())
      return !isInOut();
    return true;
  }

  /// Return true if this is a simple input-only port.  If you want the
  /// direction of the port, use the \p direction parameter.
  bool isInput() {
    if (direction != Direction::In)
      return false;
    if (type.isa<FIRRTLType>())
      return !isInOut();
    return true;
  }

  /// Return true if this is an inout port.  This will be true if the port
  /// contains either bi-directional signals or analog types.
  bool isInOut() {
    auto flags = TypeSwitch<Type, RecursiveTypeProperties>(type)
                     .Case<FIRRTLBaseType>([](auto base) {
                       return base.getRecursiveTypeProperties();
                     })
                     .Case<RefType>([](auto ref) {
                       return ref.getType().getRecursiveTypeProperties();
                     })
                     .Default([](auto) {
                       llvm_unreachable("unsupported type");
                       return RecursiveTypeProperties{};
                     });
    return !flags.isPassive || flags.containsAnalog;
  }

  /// Default constructors
  PortInfo(StringAttr name, Type type, Direction dir, StringAttr symName = {},
           Optional<Location> location = {}, Optional<AnnotationSet> annos = {})
      : name(name), type(type), direction(dir) {
    if (symName)
      sym = hw::InnerSymAttr::get(symName);
    if (location)
      loc = *location;
    if (annos)
      annotations = *annos;
  };
  PortInfo(StringAttr name, Type type, Direction dir, hw::InnerSymAttr sym,
           Location loc, AnnotationSet annos)
      : name(name), type(type), direction(dir), sym(sym), loc(loc),
        annotations(annos) {}
};

/// Verification hook for verifying module like operations.
LogicalResult verifyModuleLikeOpInterface(FModuleLike module);

class InnerSymbolOpInterface;
/// Verification hook for verifying InnerSym Attribute.
LogicalResult verifyInnerSymAttr(InnerSymbolOpInterface op);

namespace detail {
LogicalResult verifyInnerRefNamespace(Operation *op);
} // namespace detail

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

    // Verify all InnerSymbolTable's and InnerRef users.
    return ::circt::firrtl::detail::verifyInnerRefNamespace(op);
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
