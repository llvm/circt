//===- Comb.td - Comb dialect definition --------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the Comb dialect.
//
//===----------------------------------------------------------------------===//

#ifndef COMB_TD
#define COMB_TD

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

include "circt/Dialect/Comb/CombDialect.td"
include "circt/Dialect/HW/HWTypes.td"
include "circt/Dialect/Comb/Combinational.td"

#endif // COMB_TD
