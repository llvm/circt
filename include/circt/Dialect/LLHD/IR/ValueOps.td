//===- ValueOps.td - LLHD value operations -----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the MLIR ops for LLHD value creation.
//
//===----------------------------------------------------------------------===//

def LLHD_ConstOp : LLHD_Op<"const", [ConstantLike, NoSideEffect]> {
  let summary = "Introduce a new constant.";
  let description = [{
    The `llhd.const` instruction introduces a new constant value as an
    SSA-operator.
    Legal types are integers and time. Note: Signals
    are not legal to define using `llhd.const`, use the `llhd.sig`
    instruction for that.

    Syntax:

    ```
    const-op ::= ssa-id `=`
      `llhd.const` attribute-value attr-dict `:` result-type
    ```

    Example:

    ```mlir
    %0 = llhd.const 1 : i64
    %1 = llhd.const #llhd.time<1ns, 2d, 3d> : !llhd.time
    ```
  }];

  let arguments = (ins AnyAttr: $value);
  let results = (outs AnyTypeOf<[HWIntegerType, LLHD_TimeType]>: $out);

  let hasFolder = 1;
}
