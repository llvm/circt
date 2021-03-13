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

def RegOp : SeqOp<"reg",
    [NoSideEffect, AllTypesMatch<["input", "data"/*, "resetValue"*/]>,
     SameVariadicOperandSize]> {
  let summary = "Register a value, storing it for one cycle";
  let description = "See the Seq dialect rationale for a longer description";

  let arguments = (ins AnyType:$input, I1:$clk,
    Optional<I1>:$reset, Optional<AnyType>:$resetValue);
  let results = (outs AnyType:$data);

  let printer = "return ::print(p, *this);";
  let parser = "return ::parse$cppClass(parser, result);";
  // let assemblyFormat = [{
  //   $input $clk ($reset^ `(` $resetValue `)`)? attr-dict `:` custom<RegTypes>(type($input), type($resetValue))
  // }];
}

#endif // SEQ_TD
