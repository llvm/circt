//===- RTGTestOps.td - RTGTest operations ------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the RTGTest MLIR operations.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/EnumAttr.td"
include "circt/Dialect/RTG/IR/RTGInterfaces.td"

// Base class for the operation in this dialect.
class RTGTestOp<string mnemonic, list<Trait> traits = []> :
  Op<RTGTestDialect, mnemonic, traits>;

def TestInstrAOp : RTGTestOp<"instr_a", [
  DeclareOpInterfaceMethods<InstructionOpInterface>,
]> {
  let arguments = (ins I32:$imm1, I32:$imm2);
  let assemblyFormat = "$imm1 `,` $imm2 attr-dict";
}

def TestRegisterAOp : RTGTestOp<"reg_a", [
  DeclareOpInterfaceMethods<ResourceOpInterface>,
]> {
  let arguments = (ins I8Attr:$number);
  let results = (outs BogusRegType:$result);
  let assemblyFormat = "$number attr-dict";
}

def TestInstrBOp : RTGTestOp<"instr_b", [
  DeclareOpInterfaceMethods<InstructionOpInterface>,
]> {
  let arguments = (ins BogusRegType:$reg1, BogusRegType:$reg2);
  let assemblyFormat = "$reg1 `,` $reg2 attr-dict";
}
