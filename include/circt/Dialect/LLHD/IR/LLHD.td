//===- LLHD.td - LLHD dialect definition -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the LLHD dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_LLHD_IR_LLHD_TD
#define CIRCT_DIALECT_LLHD_IR_LLHD_TD

include "circt/Dialect/HW/HWTypes.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

include "circt/Dialect/LLHD/IR/LLHDDialect.td"
include "circt/Dialect/LLHD/IR/LLHDTypes.td"
include "circt/Dialect/LLHD/IR/LLHDOps.td"

#endif // CIRCT_DIALECT_LLHD_IR_LLHD_TD
