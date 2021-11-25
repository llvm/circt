//===- APInt.td - APInt dialect definition -----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the APInt dialect.
//
//===----------------------------------------------------------------------===//

#ifndef APINT_TD
#define APINT_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

def APIntDialect : Dialect {
  let name = "apint";

  let summary = "Types and operations for apint dialect";
  let description = [{
    This dialect defines the `apint` dialect, which is intended to represent
    lossless arbitrary precision integer arithmetics.
  }];
  let cppNamespace = "::circt::apint";
}

// Base class for the operation in this dialect.
class APIntOp<string mnemonic, list<OpTrait> traits = []> :
    Op<APIntDialect, mnemonic, traits>;

include "circt/Dialect/APInt/APIntTypes.td"
include "circt/Dialect/APInt/APIntOps.td"

#endif // APINT_TD
