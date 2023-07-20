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

def FormatVerilogStringOp : VerifOp<"format_verilog_string", [
    Pure
  ]> {
  let summary = "Creates a formatted string.";
  let description = [{
    Creates a formatted string suitable for printing via the `verif.print` op.
    The formatting syntax is expected to be identical to verilog string
    formatting to keep things simple for emission.
    If we in the future would like to be less tied to verilog formatting,
    please ask your friendly neighbourhood compiler engineer to e.g. implement
    a `FormatStringOp` which itself may lower to a `FormatVerilogStringOp`.
  }];
  let arguments = (ins
    StrAttr:$formatString,
    Variadic<AnyType>:$substitutions);
  
  let results = (outs HWStringType:$str);
  let assemblyFormat = [{
    $formatString `(` $substitutions `)` `:` type($substitutions) attr-dict
  }];
}

def PrintOp : VerifOp<"print", []> {
  let summary = "Prints a message.";
  let arguments = (ins HWStringType:$string);
  let assemblyFormat = [{
    $string attr-dict
  }];
}

#endif // CIRCT_DIALECT_VERIF_VERIFOPS_TD
