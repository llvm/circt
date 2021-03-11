//===- Seq.td - Seq dialect definition ---------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the Seq dialect.
//
//===----------------------------------------------------------------------===//

#ifndef SEQ_TD
#define SEQ_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def SeqDialect : Dialect {
  let name = "seq";

  let summary = "Types and operations for seq dialect";
  let description = [{
    The `seq` dialect is intended to model digital sequential logic.
  }];

  let cppNamespace = "::circt::seq";
}

// Base class for the operation in this dialect.
class SeqOp<string mnemonic, list<OpTrait> traits = []> :
    Op<SeqDialect, mnemonic, traits>;

#endif // SEQ_TD
