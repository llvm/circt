//===- RTLOps.h - Declare RTL dialect operations ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the RTL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_RTL_OPS_H
#define CIRCT_DIALECT_RTL_OPS_H

#include "circt/Dialect/RTL/RTLDialect.h"
#include "circt/Dialect/RTL/RTLTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// TODO: Remove this when OpAsmInterface qualfies this type. (To check, just
// delete this and see if it still compiles.)
namespace circt {
namespace rtl {
using mlir::OpAsmOpInterface;
using mlir::OpAsmSetValueNameFn;
} // namespace rtl
} // namespace circt

namespace circt {
namespace rtl {

/// A RTL module ports direction.
enum PortDirection {
  INPUT = 1,
  OUTPUT = 2,
  INOUT = 3,
};

/// This holds the name, type, direction of a module's ports
struct ModulePortInfo {
  mlir::StringAttr name;
  PortDirection direction;
  mlir::Type type;
  size_t argNum = ~0U; // Either the argument index or the result index
                       // depending on the direction.

  llvm::StringRef getName() const { return name ? name.getValue() : ""; }
  bool isOutput() const { return direction == OUTPUT; }
};

mlir::FunctionType getModuleType(mlir::Operation *op);

void getModulePortInfo(mlir::Operation *op,
                       llvm::SmallVectorImpl<ModulePortInfo> &results);
mlir::StringAttr getRTLNameAttr(llvm::ArrayRef<mlir::NamedAttribute> attrs);

/// Return true if the specified operation is a combinatorial logic op.
bool isCombinatorial(mlir::Operation *op);

} // namespace rtl
} // namespace circt

#define GET_OP_CLASSES
#include "circt/Dialect/RTL/RTL.h.inc"

#endif // CIRCT_DIALECT_RTL_OPS_H
