//===- ICE40Dialect.td - ICE40 dialect definition ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains the ICE40Dialect definition to be included in other files.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ICE40_ICE40DIALECT
#define CIRCT_DIALECT_ICE40_ICE40DIALECT

include "mlir/IR/OpBase.td"

def ICE40Dialect : Dialect {
  let name = "ice40";
  let cppNamespace = "::circt::ice40";

  let summary = "Types and operations for the ICE40 technology library dialect";
  let description = [{
    This dialect defines the `ice40` dialect, which is a dialect of the technology
    library for the Lattice ICE40 family of products.
  }];
}

// Base class for the operation in this dialect.
class ICE40Op<string mnemonic, list<Trait> traits = []> :
    Op<ICE40Dialect, mnemonic, traits>;

#endif // CIRCT_DIALECT_ICE40_ICE40DIALECT
