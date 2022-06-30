//===- HWArithOps.td - HW arithmetic operations ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This defines HW arithmetic operations.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpAsmInterface.td"


// Dummy op (for now) to trigger dialect document generation.
def HWArith_CastOp : HWArithOp<"cast">,
    Arguments<(ins AnyType:$in)>,
    Results<(outs AnyType:$out)> {
  let assemblyFormat = "$in attr-dict `:` type($in) `to` type($out)";
}
