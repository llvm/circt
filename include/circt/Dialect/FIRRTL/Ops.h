//===- FIRRTL/Ops.h - Declare FIRRTL dialect operations ---------*- C++ -*-===//
//
// This file declares the operation class for the FIRRTL IR.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_OPS_H
#define CIRCT_DIALECT_FIRRTL_OPS_H

#include "circt/Dialect/FIRRTL/Dialect.h"
#include "circt/Dialect/FIRRTL/Types.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace circt {
namespace firrtl {

/// Return true if the specified operation is a firrtl expression.
bool isExpression(Operation *op);

// Binary primitives.
FIRRTLType getAddSubResult(FIRRTLType lhs, FIRRTLType rhs);
FIRRTLType getMulResult(FIRRTLType lhs, FIRRTLType rhs);
FIRRTLType getDivResult(FIRRTLType lhs, FIRRTLType rhs);
FIRRTLType getRemResult(FIRRTLType lhs, FIRRTLType rhs);
FIRRTLType getCompareResult(FIRRTLType lhs, FIRRTLType rhs);
FIRRTLType getBitwiseBinaryResult(FIRRTLType lhs, FIRRTLType rhs);
FIRRTLType getCatResult(FIRRTLType lhs, FIRRTLType rhs);
FIRRTLType getDShlResult(FIRRTLType lhs, FIRRTLType rhs);
FIRRTLType getDShrResult(FIRRTLType lhs, FIRRTLType rhs);
FIRRTLType getValidIfResult(FIRRTLType lhs, FIRRTLType rhs);

// Unary primitives.
FIRRTLType getAsAsyncResetResult(FIRRTLType input);
FIRRTLType getAsClockResult(FIRRTLType input);
FIRRTLType getAsSIntResult(FIRRTLType input);
FIRRTLType getAsUIntResult(FIRRTLType input);
FIRRTLType getCvtResult(FIRRTLType input);
FIRRTLType getNegResult(FIRRTLType input);
FIRRTLType getNotResult(FIRRTLType input);
FIRRTLType getReductionResult(FIRRTLType input);
FIRRTLType getAsPassiveResult(FIRRTLType input);

typedef std::pair<StringAttr, FIRRTLType> ModulePortInfo;

/// Return the function type that corresponds to a module.
FunctionType getModuleType(Operation *op);

/// This function can extract information about ports from a module and an
/// extmodule.
void getModulePortInfo(Operation *op, SmallVectorImpl<ModulePortInfo> &results);

} // namespace firrtl
} // namespace circt

#define GET_OP_CLASSES
#include "circt/Dialect/FIRRTL/FIRRTL.h.inc"

#endif // CIRCT_DIALECT_FIRRTL_OPS_H
