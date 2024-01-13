//===- SMTBitVectorOps.td - SMT bit-vector dialect ops -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SMT_SMTBITVECTOROPS_TD
#define CIRCT_DIALECT_SMT_SMTBITVECTOROPS_TD

include "circt/Dialect/SMT/SMTDialect.td"
include "circt/Dialect/SMT/SMTAttributes.td"
include "circt/Dialect/SMT/SMTTypes.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class SMTBVOp<string mnemonic, list<Trait> traits = []> :
  Op<SMTDialect, "bv." # mnemonic, traits>;

def BVConstantOp : SMTBVOp<"constant", [
  Pure,
  ConstantLike,
  FirstAttrDerivedResultType,
  DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
]> {
  let summary = "produce a constant bit-vector";
  let description = [{
    This operation produces an SSA value equal to the bit-vector constant
    specified by the 'value' attribute.
    Refer to the `BitVectorAttr` documentation for more information about
    the semantics of bit-vector constants, their format, and associated sort.
    The result type always matches the attribute's type.

    Examples:
    ```mlir
    %bv_x5c = smt.bv.constant <"#x5c">
    %bv_b0101 = smt.bv.constant <"#b0101">
    ```
  }];

  let arguments = (ins BitVectorAttr:$value);
  let results = (outs BitVectorType:$result);

  let assemblyFormat = "$value attr-dict";

  let hasFolder = true;
  let hasVerifier = true;
}

#endif // CIRCT_DIALECT_SMT_SMTBITVECTOROPS_TD
