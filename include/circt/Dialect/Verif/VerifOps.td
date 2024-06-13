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
include "circt/Dialect/LTL/LTLDialect.td"
include "circt/Dialect/LTL/LTLTypes.td"
include "circt/Dialect/HW/HWTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/PatternBase.td"

class VerifOp<string mnemonic, list<Trait> traits = []> :
  Op<VerifDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Assertions
//===----------------------------------------------------------------------===//

class AssertLikeOp<string mnemonic, list<Trait> traits = []> :
  VerifOp<mnemonic, traits> {
  let arguments = (ins LTLAnyPropertyType:$property, Optional<I1>:$enable,
                       OptionalAttr<StrAttr>:$label);
  let assemblyFormat = [{
    $property (`if` $enable^)? (`label` $label^)? attr-dict `:` type($property)
  }];

  let hasCanonicalizeMethod = true;
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

// Edge behavior enum for always block.  See SV Spec 9.4.2.

/// AtPosEdge triggers on a rise from 0 to 1/X/Z, or X/Z to 1.
def AtPosEdge: I32EnumAttrCase<"Pos", 0, "posedge">;
/// AtNegEdge triggers on a drop from 1 to 0/X/Z, or X/Z to 0.
def AtNegEdge: I32EnumAttrCase<"Neg", 1, "negedge">;
/// AtEdge is syntactic sugar for AtPosEdge or AtNegEdge.
def AtEdge   : I32EnumAttrCase<"Both", 2, "edge">;

def ClockEdgeAttr : I32EnumAttr<"ClockEdge", "clock edge",
                                [AtPosEdge, AtNegEdge, AtEdge]> {
  let cppNamespace = "circt::verif";
}

class ClockedAssertLikeOp<string mnemonic, list<Trait> traits = []> :
  VerifOp<mnemonic, traits> {
  let arguments = (ins LTLAnyPropertyType:$property,
                      ClockEdgeAttr:$edge, I1:$clock,
                      Optional<I1>:$enable,
                      OptionalAttr<StrAttr>:$label);
  let assemblyFormat = [{
    $property (`if` $enable^)? `,` $edge $clock 
    (`label` $label^)? attr-dict `:` type($property)
  }];
}

def ClockedAssertOp : ClockedAssertLikeOp<"clocked_assert"> {
  let summary = [{ 
    Assert that a property holds, checked on a given clock's 
    ticks and enabled if a given condition holds. Only supports
    a single clock and a single disable.
  }];
}

def ClockedAssumeOp : ClockedAssertLikeOp<"clocked_assume"> {
  let summary = [{ 
    Assume that a property holds, checked on a given clock's 
    ticks and enabled if a given condition holds. Only supports
    a single clock and a single disable.
  }];
}

def ClockedCoverOp : ClockedAssertLikeOp<"clocked_cover"> {
  let summary = [{ 
    Cover on the holding of a property, checked on a given clock's 
    ticks and enabled if a given condition holds. Only supports
    a single clock and a single disable.
  }];
}

//===----------------------------------------------------------------------===//
// Printing Formatted Messages
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// Reset and Power-Cycling Detection
//===----------------------------------------------------------------------===//

def HasBeenResetOp : VerifOp<"has_been_reset", [Pure]> {
  let summary = "Check that a proper reset has been seen.";
  let description = [{
    The result of `verif.has_been_reset` reads as 0 immediately after simulation
    startup and after each power-cycle in a power-aware simulation. The result
    remains 0 before and during reset and only switches to 1 after the reset is
    deasserted again.

    This is a useful utility to disable the evaluation of assertions and other
    verification constructs in the IR before the circuit being tested has been
    properly reset. Verification failures due to uninitialized or randomized
    initial state can thus be prevented.

    Using the result of `verif.has_been_reset` to enable verification is more
    powerful and proper than just disabling verification during reset. The
    latter does not properly handle the period of time between simulation
    startup or power-cycling and the start of reset. `verif.has_been_reset` is
    guaranteed to produce a 0 value in that period, as well as during the reset.
  }];
  let arguments = (ins I1:$clock, I1:$reset, BoolAttr:$async);
  let results = (outs I1:$result);
  let assemblyFormat = [{
    $clock `,` custom<KeywordBool>($async, "\"async\"", "\"sync\"")
    $reset attr-dict
  }];
  let hasFolder = true;
}

//===----------------------------------------------------------------------===//
// Logical Equivalence Checking and Model Checking
//===----------------------------------------------------------------------===//

def LogicEquivalenceCheckingOp : VerifOp<"lec", [
  IsolatedFromAbove,
  SingleBlockImplicitTerminator<"verif::YieldOp">,
]> {
  let summary = "represents a logical equivalence checking problem";
  let description = [{
    This operation represents a logic equivalence checking problem explicitly in
    the IR. There are several possiblities to perform logical equivalence
    checking. For example, equivalence checking of combinational circuits can be
    done by constructing a miter circuit and testing whether the result is
    satisfiable (can be non-zero for some input), or two canonical BDDs could be
    constructed and compared for identity, etc.
    
    The number and types of the inputs and outputs of the two circuits (and thus
    also the block arguments and yielded values of both regions) have to match.
    Otherwise, the two should be considered trivially non-equivalent.
  }];

  let results = (outs I1:$areEquivalent);
  let regions = (region SizedRegion<1>:$firstCircuit,
                        SizedRegion<1>:$secondCircuit);

  let assemblyFormat = [{
    attr-dict `first` $firstCircuit `second` $secondCircuit
  }];

  let hasRegionVerifier = true;
}

def YieldOp : VerifOp<"yield", [
  Terminator,
  ParentOneOf<["verif::LogicEquivalenceCheckingOp"]>
]> {
  let summary = "yields values from a region";

  let arguments = (ins Variadic<AnyType>:$inputs);
  let assemblyFormat = "($inputs^ `:` type($inputs))? attr-dict";

  let builders = [
    OpBuilder<(ins), [{
      build($_builder, $_state, std::nullopt);
    }]>,
  ];
}

#endif // CIRCT_DIALECT_VERIF_VERIFOPS_TD
