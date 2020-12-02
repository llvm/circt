//===- SV/Ops.h - Declare SV dialect operations -----------------*- C++ -*-===//
//
// This file declares the operation classes for the SV dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SV_OPS_H
#define CIRCT_DIALECT_SV_OPS_H

#include "circt/Dialect/SV/Dialect.h"
#include "circt/Dialect/SV/Types.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "circt/Dialect/SV/SV.h.inc"
#include "circt/Dialect/SV/SVEnums.h.inc"
#include "circt/Dialect/SV/SVStructs.h.inc"

#endif // CIRCT_DIALECT_SV_OPS_H
