//===- LLHDValueOps.td - LLHD value operations -------------*- tablegen -*-===//
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

include "circt/Dialect/LLHD/IR/LLHDDialect.td"
include "circt/Dialect/LLHD/IR/LLHDTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def ConstantTimeOp : LLHDOp<"constant_time",
                                  [ConstantLike, Pure]> {
  let summary = "Introduce a new time constant.";
  let description = [{
    The `llhd.constant_time` instruction introduces a new constant time value as
    an SSA-operator.

    Example:

    ```mlir
    %1 = llhd.constant_time #llhd.time<1ns, 2d, 3d>
    ```
  }];

  let assemblyFormat = "$value attr-dict";

  let builders = [
    /// Build a ConstantTimeOp from an ArrayRef with three values
    /// (real-time, delta, epsilon) and a string for the real-time unit.
    OpBuilder<(ins "unsigned":$time,
                   "const StringRef &":$timeUnit,
                   "unsigned":$delta,
                   "unsigned":$epsilon)>
  ];

  let arguments = (ins LLHDTimeAttr: $value);
  let results = (outs LLHDTimeType: $result);

  let hasFolder = 1;
}
