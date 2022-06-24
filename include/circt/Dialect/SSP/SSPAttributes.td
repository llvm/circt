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

include "SSPPropertyInterfaces.td"

def DependenceAttr : SSPAttribute<"Dependence"> {
  let description = [{
    An attribute to uniformly model def-use and auxiliary
    [dependences](https://circt.llvm.org/docs/Scheduling/#components) as well as
    to attach [properties](https://circt.llvm.org/docs/Scheduling/#properties)
    to them. This attribute is an implementation detail of the `ssp.OperationOp`
    and as such is supposed to be hidden by the custom parser/printer.
  }];

  let parameters = (ins OptionalParameter<"::mlir::IntegerAttr">:$operandIdx,
                        OptionalParameter<"::mlir::FlatSymbolRefAttr">:$sourceRef,
                        "::mlir::DictionaryAttr":$properties);

  let mnemonic = "dependence";
  let assemblyFormat = [{
    `<` (`defuse` $operandIdx^)? (`auxiliary` $sourceRef^)? `,` $properties `>`
  }];
  let genVerifyDecl = 1;
}

def DependenceArrayAttr
  : TypedArrayAttrBase<DependenceAttr, "dependence array attribute">;
