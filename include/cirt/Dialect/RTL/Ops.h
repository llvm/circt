//===- RTL/Ops.h - Declare RTL dialect operations ---------------*- C++ -*-===//
//
// This file declares the operation classes for the RTL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRT_DIALECT_RTL_OPS_H
#define CIRT_DIALECT_RTL_OPS_H

#include "cirt/Dialect/RTL/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace cirt {
namespace rtl {

#define GET_OP_CLASSES
#include "cirt/Dialect/RTL/RTL.h.inc"

/// Return true if the specified operation is a combinatorial logic op.
bool isCombinatorial(Operation *op);

} // namespace rtl
} // namespace cirt

#endif // CIRT_DIALECT_RTL_OPS_H
