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

def DependenceAttr : SSPAttribute<"Dependence"> {
  let summary = "TODO";
  let description = [{ TODO }];
  let mnemonic = "dependence";
  
  let parameters = (ins OptionalParameter<"::mlir::IntegerAttr">:$operandIdx,
                        OptionalParameter<"::mlir::FlatSymbolRefAttr">:$sourceRef,
                        "::mlir::DictionaryAttr":$properties);
  // This format is not meant to be user-facing. The `ssa` and `aux` clauses
  // are mutually exclusive.
  let assemblyFormat = [{
    `<` (`ssa` $operandIdx^)? (`aux` $sourceRef^)? `,` $properties `>`
  }];
  let genVerifyDecl = 1;
}

def DependenceArrayAttr
  : TypedArrayAttrBase<DependenceAttr, "dependence array attribute">;
