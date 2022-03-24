//===- MIRExpressions.td - Moore MIR expression ops --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the ops for Moore MIR expressions.
//
//===----------------------------------------------------------------------===//

include "mlir/Interfaces/InferTypeOpInterface.td"


def ConstantOp : MIROp<"constant", [NoSideEffect]> {
  let summary = "A constant value";

  let arguments = (ins I32Attr:$value);
  let results = (outs MooreRValueType:$result);
  let assemblyFormat = "$value attr-dict `:` qualified(type($result))";
}

def ConcatOp : MIROp<"concat", [
    NoSideEffect, DeclareOpInterfaceMethods<InferTypeOpInterface>
]> {
  let summary = "A concatenation of expressions";
  let description = [{
    This operation represents the SystemVerilog concatenation expression
    `{x, y, z}`. See IEEE 1800-2017 §11.4.12 "Concatenation operators".

    All operands must be simple bit vector types.

    The concatenation result is a simple bit vector type. The result is unsigned
    regardless of the sign of the operands (see concatenation-specific rules in
    IEEE 1800-2017 §11.8.1 "Rules for expression types"). The size of the result
    is the sum of the sizes of all operands. If any of the operands is
    four-valued, the result is four-valued; otherwise it is two-valued.
  }];
  let arguments = (ins Variadic<SimpleBitVectorType>:$values);
  let results = (outs SimpleBitVectorType:$result);
  let assemblyFormat = [{
    $values attr-dict `:` functional-type($values, $result)
  }];
}
