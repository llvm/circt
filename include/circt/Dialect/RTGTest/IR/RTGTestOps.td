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

include "mlir/IR/CommonAttrConstraints.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "circt/Dialect/RTG/IR/RTGInterfaces.td"
include "circt/Dialect/RTG/IR/RTGISAAssemblyInterfaces.td"

// Base class for the operation in this dialect.
class RTGTestOp<string mnemonic, list<Trait> traits = []> :
  Op<RTGTestDialect, mnemonic, traits>;


def CPUDeclOp : RTGTestOp<"cpu_decl", [
  Pure,
  ConstantLike,
  ContextResourceDefining,
  FirstAttrDerivedResultType,
]> {
  let summary = "declare a CPU";
  let description = [{
    This operation is used to test the `ContextResourceOpInterface` and passes
    taking advantage of it.
  }];

  let arguments = (ins CPUAttr:$id);
  let results = (outs CPUType:$cpu);

  let assemblyFormat = "$id attr-dict";
  let hasFolder = 1;
}

def ConstantTestOp : RTGTestOp<"constant_test", [
  Pure, ConstantLike,
]> {
  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType:$result);

  let assemblyFormat = "type($result) attr-dict";
  let hasFolder = 1;
}

def RegisterOp : RTGTestOp<"reg", [
  DeclareOpInterfaceMethods<RegisterOpInterface>,
  DeclareOpInterfaceMethods<InferTypeOpInterface>,
]> {
  let summary = "returns a value representing a register";
  let description = [{
    This operation creates a value representing the register given as the 'reg'
    attribute. A register can be a concrete register or a virtual register.
    Virtual registers will be assigned a concrete register when running register
    allocation.
  }];

  let arguments = (ins RegisterAttr:$reg);
  let results = (outs RegisterType:$result);

  let assemblyFormat = "$reg attr-dict";
}
