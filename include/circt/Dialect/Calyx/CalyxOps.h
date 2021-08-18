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
#include "mlir/IR/OpDefinition.h"
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

/// The direction of a Calyx port.
enum PortDirection { INPUT = 0, OUTPUT = 1 };

/// This holds the name and type that describes the component's ports.
struct ComponentPortInfo {
  StringAttr name;
  Type type;
  PortDirection direction;
};

/// A helper function to verify each operation with the Cell trait.
LogicalResult verifyCell(Operation *op);

/// Signals that the following operation is a cell.
template <typename ConcreteType>
class Cell : public mlir::OpTrait::TraitBase<ConcreteType, Cell> {
public:
  static LogicalResult verifyTrait(Operation *op) { return verifyCell(op); }
  SmallVector<StringRef> portNames();
  SmallVector<PortDirection> portDirections();
};

/// Returns port information about a given component.
SmallVector<ComponentPortInfo> getComponentPortInfo(Operation *op);

} // namespace calyx
} // namespace circt

#define GET_OP_CLASSES
#include "circt/Dialect/Calyx/Calyx.h.inc"

#endif // CIRCT_DIALECT_CALYX_OPS_H
