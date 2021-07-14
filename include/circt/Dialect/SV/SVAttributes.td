//===- SVAttributes.td - SV attribute definitions ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

class SVAttr<string name> : AttrDef<SVDialect, name> { }

def VerbatimAttr : SVAttr<"VerbatimParameter"> {
  let summary = "Represents text to emit directly to SystemVerilog";
  let parameters = (ins StringRefParameter<"">:$value);
  let mnemonic = "verbatim.parameter";
}
