//===- Sim.td - Sim dialect definition ---------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the Sim dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SIM_SIM_TD
#define CIRCT_DIALECT_SIM_SIM_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

include "circt/Dialect/Sim/SimDialect.td"
include "circt/Dialect/Sim/SimOps.td"

#endif // CIRCT_DIALECT_SIM_SIM_TD
