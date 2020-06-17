//===- RTL/Ops.h - Declare RTL dialect operations ---------------*- C++ -*-===//
//
// This file declares the operation classes for the RTL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_RTL_OPS_H
#define CIRCT_DIALECT_RTL_OPS_H

#include "circt/Dialect/RTL/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace circt {
namespace rtl {

#define GET_OP_CLASSES
#include "circt/Dialect/RTL/RTL.h.inc"

/// Return true if the specified operation is a combinatorial logic op.
bool isCombinatorial(Operation *op);

} // namespace rtl
} // namespace circt

#endif // CIRCT_DIALECT_RTL_OPS_H
