//===- Interop.td - Interop dialect definition -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the Interop dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_INTEROP_INTEROP
#define CIRCT_DIALECT_INTEROP_INTEROP

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Interop Dialect
//===----------------------------------------------------------------------===//

def InteropDialect : Dialect {
  let name = "interop";

  let summary = "Provides interoperability between backends and tools";
  let description = [{
    This dialect defines the `interop` dialect which defines operations and
    interfaces necessary to provide interoperability between backends and
    and external tools without the need of writing custom pairwise interop
    solutions.
  }];
  let cppNamespace = "::circt::interop";
}

//===----------------------------------------------------------------------===//
// Interop Operations
//===----------------------------------------------------------------------===//

// Base class for the operations in this dialect.
class InteropOp<string mnemonic, list<Trait> traits = []> :
  Op<InteropDialect, mnemonic, traits>;

#endif // CIRCT_DIALECT_INTEROP_INTEROP
