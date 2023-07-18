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
include "circt/Dialect/HW/HWTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

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

def FormatStringOp : VerifOp<"fstr", [
    Pure
  ]> {
  let summary = "Creates a formatted string.";
  let description = [{
    Creates a formatted string suitable for printing via the `verif.printf` op.
    The formatting syntax is (for now) expected to be identical to verilog
    string formatting to keep things simple for emission. This should change
    once this op is expected to be used for more than just emitting sv.
  }];
  let arguments = (ins
    StrAttr:$formatString,
    Variadic<AnyType>:$substitutions);
  
  let results = (outs HWStringType:$str);
  let assemblyFormat = [{
    $formatString `(` $substitutions `)` `:` type($substitutions) attr-dict
  }];
}

def PrintfOp : VerifOp<"printf", []> {
  let summary = "Prints a formatted message.";
  let arguments = (ins HWStringType:$formattedString);
  let assemblyFormat = [{
    $formattedString attr-dict
  }];
}

#endif // CIRCT_DIALECT_VERIF_VERIFOPS_TD
