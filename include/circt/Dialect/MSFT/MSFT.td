//===- MSFT.td - MSFT dialect definition -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the MSFT dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MSFT_TD
#define MSFT_TD

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Pass/PassBase.td"

include "circt/Dialect/MSFT/MSFTDialect.td"
include "circt/Dialect/MSFT/MSFTAttributes.td"
include "circt/Dialect/MSFT/MSFTOpInterfaces.td"
include "circt/Dialect/MSFT/MSFTPDOps.td"
include "circt/Dialect/MSFT/MSFTConstructs.td"
include "circt/Dialect/MSFT/MSFTPasses.td"

#endif // MSFT_TD
