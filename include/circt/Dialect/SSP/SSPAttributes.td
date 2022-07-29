//===- SSPAttributes.td - SSP attribute definitions --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the SSP (static scheduling problem) dialect attributes.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Dialect attributes
//===----------------------------------------------------------------------===//

def DependenceAttr : AttrDef<SSPDialect, "Dependence"> {
  let summary = "Internal representation of dependence edges.";
  let description = [{
    An attribute to uniformly model def-use and auxiliary
    [dependences](https://circt.llvm.org/docs/Scheduling/#components) as well as
    to attach [properties](https://circt.llvm.org/docs/Scheduling/#properties)
    to them. This attribute is an implementation detail of the `ssp.OperationOp`
    and as such is supposed to be hidden by the custom parser/printer.
  }];

  let parameters = (ins "unsigned":$operandIdx,
                        OptionalParameter<"::mlir::FlatSymbolRefAttr">:$sourceRef,
                        OptionalParameter<"::mlir::ArrayAttr">:$properties);

  let mnemonic = "dependence";
  let assemblyFormat = [{
    `<` $operandIdx (`,` $sourceRef^)? (`:` $properties^)? `>`
  }];
}

def DependenceArrayAttr
  : TypedArrayAttrBase<DependenceAttr, "dependence array attribute">;

//===----------------------------------------------------------------------===//
// Property definitions for in-tree problems
//===----------------------------------------------------------------------===//

// Real definition to follow in next PR
def DummyLinkedOperatorType : AttrDef<SSPDialect, "LinkedOperatorType"> {
  let parameters = (ins "::mlir::FlatSymbolRefAttr":$value);
  let mnemonic = "dummy_opr";
  let assemblyFormat = "`<` $value `>`";
}
