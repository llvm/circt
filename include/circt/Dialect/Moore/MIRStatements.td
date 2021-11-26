//===- MIRStatements.td - Moore MIR statements ops ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the ops for Moore MIR statements.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Assignment Statements
//===----------------------------------------------------------------------===//

def AssignOp : MIROp<"assign", []> {
  let summary = "Continuous assignment";
  let description = [{
    A SystemVerilog assignment statement 'x = y;'.
    These occur in module scope.  See SV Spec 10.3.2.
  }];
  let arguments = (ins MooreLValueType:$dest, MooreRValueType:$src);
  let results = (outs);
  let assemblyFormat = [{
    $dest `,` $src  attr-dict `:` type($dest) `,` type($src)
  }];
}
