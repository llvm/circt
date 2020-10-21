//===- RTL/Ops.h - Declare RTL dialect operations ---------------*- C++ -*-===//
//
// This file declares the operation classes for the RTL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_RTL_OPS_H
#define CIRCT_DIALECT_RTL_OPS_H

#include "circt/Dialect/RTL/Dialect.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace circt {
namespace rtl {

/// A RTL module ports direction.
enum PortDirection {
  INPUT = 1,
  OUTPUT = 2,
  INOUT = 3,
};

/// This holds the name, type, direction of a module's ports
struct RTLModulePortInfo {
  StringAttr name;
  PortDirection direction;
  Type type;
  size_t argNum = ~0U; // Either the argument index or the result index
                       // depending on the direction.

  inline StringRef getName() { return name ? name.getValue() : ""; }
};

FunctionType getModuleType(Operation *op);

void getRTLModulePortInfo(Operation *op,
                          SmallVectorImpl<RTLModulePortInfo> &results);
StringAttr getRTLNameAttr(ArrayRef<NamedAttribute> attrs);

/// Return true if the specified operation is a combinatorial logic op.
bool isCombinatorial(Operation *op);

} // namespace rtl
} // namespace circt

#define GET_OP_CLASSES
#include "circt/Dialect/RTL/RTL.h.inc"

#endif // CIRCT_DIALECT_RTL_OPS_H
