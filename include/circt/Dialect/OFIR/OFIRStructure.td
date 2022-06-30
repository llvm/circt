//===- OFIRStructure.td - Circuit and Module Ops -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the OFIR Class and Module MLIR ops.
//
//===----------------------------------------------------------------------===//

def ClassOp : OFIROp<"class", [IsolatedFromAbove, Symbol, SingleBlock,
NoTerminator, HasParent<"firrtl::CircuitOp">]> {
  let summary = "OFIR Class";
  let  description = [{
    The "ofir.class" operation represents a class in OFIR, including a list of
   properties, subtypeing relationships, and inheritance.
  }];
  let arguments = (ins StrAttr:$sym_name, ArrayAttr:$subTypeOf);
  let results = (outs);
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;

}

