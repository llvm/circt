//===- SystemCStructure.td - SystemC structure ops ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes ops with regions for SystemC to structure other ops, such as
// modules, methods, etc.
//
//===----------------------------------------------------------------------===//


def CtorOp : SystemCOp<"ctor", [SingleBlock, NoTerminator]> {
  let summary = "A constructor definition.";
  let description = [{
    Represents the SC_CTOR macro as described in IEEE 1666-2011 §5.2.7.
    The name of the module being constructed does not have to be passed
    to this operation, but is automatically added during emission.
  }];

  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$body attr-dict";
}
