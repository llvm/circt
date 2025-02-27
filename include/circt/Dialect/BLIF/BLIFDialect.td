//===- BLIFDialect.td - BLIF dialect definition ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains the BLIFDialect definition to be included in other files.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_BLIF_BLIFDIALECT
#define CIRCT_DIALECT_BLIF_BLIFDIALECT

include "mlir/IR/OpBase.td"

def BLIFDialect : Dialect {
  let name = "blif";
  let cppNamespace = "::circt::blif";

  let summary = "Types and operations for the BLIF format dialect";
  let description = [{
    This dialect defines the `BLIF` dialect, which is a representation of
    the Berkely Logic Interchange Format.
  }];
}

// Base class for the operation in this dialect.
class BLIFOp<string mnemonic, list<Trait> traits = []> :
    Op<BLIFDialect, mnemonic, traits>;

#endif // CIRCT_DIALECT_BLIF_BLIFDIALECT
