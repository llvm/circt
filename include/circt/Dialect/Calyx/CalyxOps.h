//===- CalyxOps.h - Declare Calyx dialect operations ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation class for the Calyx IR.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_CALYX_OPS_H
#define CIRCT_DIALECT_CALYX_OPS_H

#include "circt/Dialect/Calyx/CalyxDialect.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace circt {
namespace calyx {

/// A helper function to verify each control-like operation
/// has a valid parent and, if applicable, body.
LogicalResult verifyControlLikeOp(Operation *op);

/// Signals that the following operation is "control-like."
template <typename ConcreteType>
class ControlLike : public mlir::OpTrait::TraitBase<ConcreteType, ControlLike> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return verifyControlLikeOp(op);
  }
};

/// A helper function to verify a combinational operation.
LogicalResult verifyCombinationalOp(Operation *op);

/// Signals that the following operation is combinational.
template <typename ConcreteType>
class Combinational
    : public mlir::OpTrait::TraitBase<ConcreteType, Combinational> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return verifyCombinationalOp(op);
  }
};

/// The port direction attribute follows the implementation style of FIRRTL
/// module port direction attributes.
enum Direction { Input = 0, Output };
namespace direction {

/// The key in a components's attribute dictionary used to find the direction.
constexpr const char *attrKey = "portDirections";

/// Return an output direction if \p isOutput is true, otherwise return an
/// input direction.
Direction get(bool isOutput);

/// Return a \p IntegerAttr containing the packed representation of an array
/// of directions.
IntegerAttr packAttribute(ArrayRef<Direction> a, MLIRContext *b);

/// Turn a packed representation of port attributes into a vector that can be
/// worked with.
SmallVector<Direction> unpackAttribute(Operation *component);

/// Convenience function for generating a vector of directions.
SmallVector<Direction> genInOutDirections(size_t nIns, size_t nOuts);
} // namespace direction

/// This holds information about the port to either a component or cell.
struct PortInfo {
  StringAttr name;
  Type type;
  Direction direction;
  DictionaryAttr attributes;

  /// Returns whether the given port has attribute with Identifier `name`.
  bool hasAttribute(StringRef identifier) const {
    assert(attributes && "PortInfo::attributes should be instantiated.");
    return llvm::any_of(attributes, [&](auto idToAttribute) {
      return identifier == std::get<0>(idToAttribute);
    });
  }

  /// Returns the attribute associated with the given name if it exists,
  /// otherwise std::nullopt.
  llvm::Optional<Attribute> getAttribute(StringRef identifier) const {
    assert(attributes && "PortInfo::attributes should be instantiated.");
    auto it = llvm::find_if(attributes, [&](auto idToAttribute) {
      return identifier == std::get<0>(idToAttribute);
    });
    if (it == attributes.end())
      return None;
    return std::get<1>(*it);
  }

  /// Returns all identifiers for this dictionary attribute.
  SmallVector<StringRef> getAllIdentifiers() const {
    assert(attributes && "PortInfo::attributes should be instantiated.");
    SmallVector<StringRef> identifiers;
    llvm::transform(
        attributes, std::back_inserter(identifiers),
        [](auto idToAttribute) { return std::get<0>(idToAttribute); });
    return identifiers;
  }
};

/// A helper function to verify each operation with the Cell trait.
LogicalResult verifyCell(Operation *op);

/// A helper function to verify each operation with the Group Interface trait.
LogicalResult verifyGroupInterface(Operation *op);

/// Returns port information for the block argument provided.
PortInfo getPortInfo(BlockArgument arg);

} // namespace calyx
} // namespace circt

#include "circt/Dialect/Calyx/CalyxInterfaces.h.inc"

#define GET_OP_CLASSES
#include "circt/Dialect/Calyx/Calyx.h.inc"

#endif // CIRCT_DIALECT_CALYX_OPS_H
