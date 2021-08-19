//===- HWOps.h - Declare HW dialect operations ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the HW dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_OPS_H
#define CIRCT_DIALECT_HW_OPS_H

#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "circt/Dialect/HW/HWOpInterfaces.h.inc"

namespace circt {
namespace hw {

/// A HW module ports direction.
enum PortDirection {
  INPUT = 1,
  OUTPUT = 2,
  INOUT = 3,
};

/// This holds the name, type, direction of a module's ports
struct ModulePortInfo {
  StringAttr name;
  PortDirection direction;
  Type type;
  size_t argNum = ~0U; // Either the argument index or the result index
                       // depending on the direction.

  StringRef getName() const { return name.getValue(); }
  bool isOutput() const { return direction == OUTPUT; }
};

// Helpers for working with modules.

/// Return true if this is an hw.module, external module, generated module etc.
bool isAnyModule(Operation *module);

/// Return the signature for the specified module as a function type.
FunctionType getModuleType(Operation *module);

/// Returns the verilog module name attribute or symbol name of any module-like
/// operations.
StringAttr getVerilogModuleNameAttr(Operation *module);

/// Return the port name for the specified argument or result.  These can only
/// return a null StringAttr when the IR is invalid.
StringAttr getModuleArgumentNameAttr(Operation *module, size_t argNo);
StringAttr getModuleResultNameAttr(Operation *module, size_t argNo);

static inline StringRef getModuleArgumentName(Operation *module, size_t argNo) {
  auto attr = getModuleArgumentNameAttr(module, argNo);
  return attr ? attr.getValue() : StringRef();
}
static inline StringRef getModuleResultName(Operation *module,
                                            size_t resultNo) {
  auto attr = getModuleResultNameAttr(module, resultNo);
  return attr ? attr.getValue() : StringRef();
}

void setModuleArgumentNames(Operation *module, ArrayRef<Attribute> names);
void setModuleResultNames(Operation *module, ArrayRef<Attribute> names);

/// Return an encapsulated set of information about input and output ports.
SmallVector<ModulePortInfo> getModulePortInfo(Operation *op);

/// Return true if the specified operation is a combinatorial logic op.
bool isCombinatorial(Operation *op);

} // namespace hw
} // namespace circt

#define GET_OP_CLASSES
#include "circt/Dialect/HW/HW.h.inc"

#endif // CIRCT_DIALECT_HW_OPS_H
