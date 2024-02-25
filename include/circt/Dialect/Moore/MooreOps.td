//===- MooreOps.td - Moore dialect operations --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MOORE_MOOREOPS
#define CIRCT_DIALECT_MOORE_MOOREOPS

include "circt/Dialect/Moore/MooreDialect.td"
include "circt/Dialect/Moore/MooreTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for the operations in this dialect.
class MooreOp<string mnemonic, list<Trait> traits = []> :
  Op<MooreDialect, mnemonic, traits>;

// Base class for the MIR operations in this dialect.
class MIROp<string mnemonic, list<Trait> traits = []> :
  MooreOp<"mir." # mnemonic, traits>;

include "circt/Dialect/Moore/MIRExpressions.td"
include "circt/Dialect/Moore/MIRStatements.td"

#endif // CIRCT_DIALECT_MOORE_MOOREOPS
