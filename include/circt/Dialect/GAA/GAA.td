//===- GAA.td - GAA dialect definition ---------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the Guard-Atomic-Action dialect.
//
//===----------------------------------------------------------------------===//
#ifndef CIRCT_DIALECT_GAA_TD
#define CIRCT_DIALECT_GAA_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
// FunctionOpInterface
include "mlir/IR/FunctionInterfaces.td"
// OpInterface
include "mlir/Interfaces/InferTypeOpInterface.td"
// Symbol, SymbolTable
include "mlir/IR/SymbolInterfaces.td"
// ReturnLike
include "mlir/Interfaces/ControlFlowInterfaces.td"
// HWIntegerType
include "circt/Dialect/HW/HWTypes.td"
// HWModuleLike
include "circt/Dialect/HW/HWOpInterfaces.td"

include "GAAOpInterfaces.td"
include "GAADialect.td"
include "GAAAttributes.td"
include "GAATypes.td"
include "GAAOps.td"

#endif // CIRCT_DIALECT_GAA_TD