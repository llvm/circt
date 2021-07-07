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
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace circt {
namespace calyx {

/// The direction of a Calyx port.
enum PortDirection { INPUT = 0, OUTPUT = 1 };

/// This holds the name and type that describes the component's ports.
struct ComponentPortInfo {
  StringAttr name;
  Type type;
  PortDirection direction;
};

/// Returns port information about a given component.
SmallVector<ComponentPortInfo> getComponentPortInfo(Operation *op);

} // namespace calyx
} // namespace circt

#define GET_OP_CLASSES
#include "circt/Dialect/Calyx/Calyx.h.inc"

#endif // CIRCT_DIALECT_CALYX_OPS_H
