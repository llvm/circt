//===- FIRRTL/IR/Ops.h - Declare FIRRTL dialect operations ------*- C++ -*-===//
//
// This file declares the operation class for the FIRRTL IR.
//
//===----------------------------------------------------------------------===//

#ifndef SPT_DIALECT_FIRRTL_IR_OPS_H
#define SPT_DIALECT_FIRRTL_IR_OPS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffects.h"
#include "spt/Dialect/FIRRTL/IR/Dialect.h"
#include "spt/Dialect/FIRRTL/IR/Types.h"

namespace spt {
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

typedef std::pair<StringAttr, FIRRTLType> ModulePortInfo;

/// Return the function type that corresponds to a module.
FunctionType getModuleType(Operation *op);

/// This function can extract information about ports from a module and an
/// extmodule.
void getModulePortInfo(Operation *op, SmallVectorImpl<ModulePortInfo> &results);

} // namespace firrtl
} // namespace spt

namespace spt {
namespace firrtl {

#define GET_OP_CLASSES
#include "spt/Dialect/FIRRTL/IR/FIRRTL.h.inc"

} // namespace firrtl
} // namespace spt

#endif // SPT_DIALECT_FIRRTL_IR_OPS_H
