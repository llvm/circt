//===- VerifOps.td - Verif dialect operations --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_VERIF_VERIFOPS_TD
#define CIRCT_DIALECT_VERIF_VERIFOPS_TD

include "circt/Dialect/Verif/VerifDialect.td"
include "circt/Dialect/LTL/LTLTypes.td"

class VerifOp<string mnemonic, list<Trait> traits = []> :
  Op<VerifDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Assertions
//===----------------------------------------------------------------------===//

class AssertLikeOp<string mnemonic, list<Trait> traits = []> :
  VerifOp<mnemonic, traits> {
  let arguments = (ins LTLAnyPropertyType:$property,
                       OptionalAttr<StrAttr>:$label);
  let assemblyFormat = [{
    $property (`label` $label^)? attr-dict `:` type($property)
  }];
}

def AssertOp : AssertLikeOp<"assert"> {
  let summary = "Assert that a property holds.";
}

def AssumeOp : AssertLikeOp<"assume"> {
  let summary = "Assume that a property holds.";
}

def CoverOp : AssertLikeOp<"cover"> {
  let summary = "Ensure that a property can hold.";
}

#endif // CIRCT_DIALECT_VERIF_VERIFOPS_TD
