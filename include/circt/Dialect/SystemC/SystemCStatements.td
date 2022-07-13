//===- SystemCStatements.td - SystemC statements ops -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the ops for SystemC statements.
//
//===----------------------------------------------------------------------===//


def CtorOp : SystemCOp<"ctor", [SingleBlock, NoTerminator]> {
  let summary = "A constructor definition.";

  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$body attr-dict";
}
