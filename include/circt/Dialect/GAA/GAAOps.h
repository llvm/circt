//===- GAAOps.h - GAA Dialect Operators -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the GAA dialect operators.
//
//===----------------------------------------------------------------------===//
#ifndef CIRCT_DIALECT_GAA_GAAOPS_H
#define CIRCT_DIALECT_GAA_GAAOPS_H

#include "llvm/ADT/Any.h"

// provides implementations for FunctionInterface.td
#include "mlir/IR/FunctionInterfaces.h"
// LogicalResult
#include "mlir/IR/Diagnostics.h"
// provides implementations for OpAsmInterface.td
#include "mlir/IR/OpImplementation.h"
// provides implementations for SymbolInterfaces.td
#include "mlir/IR/SymbolTable.h"
// provides implementations for ControlFlowInterfaces.td
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/IR/Dialect.h"

#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/GAA/GAAOpInterfaces.h"
#define GET_OP_CLASSES
#include "circt/Dialect/GAA/GAA.h.inc"
namespace circt {
namespace gaa {
GAAModuleLike getReferenceModule(InstanceOp instance);
llvm::SmallVector<InstanceOp, 4> getInstances(GAAModuleLike module);
llvm::SmallVector<GAAFunctionLike, 4> getFunctions(GAAModuleLike module);
llvm::SmallVector<MethodOp, 4> getMethods(ModuleOp module);
llvm::SmallVector<BindMethodOp, 4> getMethods(ExtModuleOp module);
llvm::SmallVector<ValueOp, 4> getValues(ModuleOp module);
llvm::SmallVector<BindValueOp, 4> getValues(ExtModuleOp module);
llvm::SmallVector<RuleOp, 4> getRules(ModuleOp module);
}
}

#endif // CIRCT_DIALECT_GAA_GAAOPS_H
