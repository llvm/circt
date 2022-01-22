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
