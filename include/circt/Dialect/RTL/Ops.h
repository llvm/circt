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
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace circt {
namespace rtl {

// This holds the name, type, direction of a module's ports
struct RTLModulePortInfo {
  StringAttr name;
  Type type;
  StringAttr direction;
  Type t2;
};

// typedef std::tuple<StringAttr, Type, StringAttr> RTLModulePortInfo;

FunctionType getModuleType(Operation *op);

void getRTLModulePortInfo(Operation *op,
                          SmallVectorImpl<RTLModulePortInfo> &results);
StringAttr getRTLNameAttr(ArrayRef<NamedAttribute> attrs);
StringAttr getRTLDirectionAttr(ArrayRef<NamedAttribute> attrs);

/// Return true if the specified operation is a combinatorial logic op.
bool isCombinatorial(Operation *op);

} // namespace rtl
} // namespace circt

#define GET_OP_CLASSES
#include "circt/Dialect/RTL/RTL.h.inc"

#endif // CIRCT_DIALECT_RTL_OPS_H
