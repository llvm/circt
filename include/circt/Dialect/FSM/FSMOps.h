//===- FSMOps.h - FSM dialect operations declaration file -----------------===//
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FSM_FSMOPS_H
#define CIRCT_DIALECT_FSM_FSMOPS_H

#include "circt/Dialect/FSM/FSMDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

#define GET_OP_CLASSES
#include "circt/Dialect/FSM/FSM.h.inc"

#endif // CIRCT_DIALECT_FSM_FSMOPS_H
