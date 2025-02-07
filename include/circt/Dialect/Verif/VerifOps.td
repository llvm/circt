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
include "circt/Dialect/Verif/VerifOpInterfaces.td"
include "circt/Dialect/LTL/LTLDialect.td"
include "circt/Dialect/LTL/LTLTypes.td"
include "circt/Dialect/HW/HWTypes.td"
include "circt/Dialect/Seq/SeqTypes.td"
include "circt/Dialect/HW/HWAttributes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/PatternBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"

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

def BoundedModelCheckingOp : VerifOp<"bmc", [
  IsolatedFromAbove,
  SingleBlockImplicitTerminator<"verif::YieldOp">,
]> {
  let summary = "Perform a bounded model check";
  let description = [{
    This operation represents a bounded model checking problem explicitly in
    the IR. The `bound` attribute indicates how many times the `circuit` region
    should be executed, and `num_regs` indicates the number of registers in the
    design that have been externalized and appended to the region's
    inputs/outputs (these values are fed from each `circuit` region execution
    to the next, as they represent register state, rather than being
    overwritten with fresh variables like other inputs). `initial_values` is an
    array containing the initial value of each register - where the register
    has no initial value, a unit attribute is given. The `circuit` region
    contains the circuit (alongside the `verif` property checking operations)
    to be checked.

    The `init` region contains the logic to initialize the clock signals, and
    will be executed once before any other region - it cannot take any
    arguments, and should return as many `!seq.clock` values as the `circuit`
    region has `!seq.clock` arguments, followed by any initial arguments of
    'state' arguments to be fed to the `loop` region (see below).

    The `loop` region contains the logic to advance the clock signals, and will
    be executed after each execution of the `circuit` region. It should take as
    arguments as many `!seq.clock` values as the `circuit` region has, and
    these can be followed by additional 'state' arguments to represent e.g.
    which clock should be toggled next. The types yielded should be the same,
    as this region yields the updated clock and state values (this should also
    match the types yielded by the `init` region).
  }];

  // TODO: initial values should eventually be handled by init region
  let arguments = (ins I32Attr:$bound, I32Attr:$num_regs, ArrayAttr:$initial_values);
  let regions = (region SizedRegion<1>:$init,
                        SizedRegion<1>:$loop,
                        SizedRegion<1>:$circuit);

  let results = (outs I1:$result);

  let assemblyFormat = [{
    `bound` $bound `num_regs` $num_regs `initial_values` $initial_values attr-dict-with-keyword `init` $init `loop` $loop `circuit`
    $circuit
  }];

  let hasRegionVerifier = true;
}

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
  ParentOneOf<[
    "verif::LogicEquivalenceCheckingOp", "verif::BoundedModelCheckingOp",
    "verif::ContractOp"
  ]>
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

//===----------------------------------------------------------------------===//
// Formal Test
//===----------------------------------------------------------------------===//

def FormalOp : VerifOp<"formal", [
  IsolatedFromAbove,
  NoTerminator,
  RegionKindInterface,
  SingleBlock,
  Symbol,
]> {
  let summary = "A formal unit test";
  let description = [{
    This operation defines a formal unit test that can be automatically run by
    various tools. To describe a test, the body of this op should contain the
    hardware to be tested, alongside any asserts, assumes, and covers to be
    formally verified. The body can contain instances of other modules, in which
    case all asserts, assumes, and covers in those modules are also verified.

    The `verif.symbolic_value` op can be used to create symbolic values to feed
    into the hardware to be tested. Testing tools will then try to find concrete
    values for them that violate any asserts or make any covers true.

    ### Example
    ```
    verif.formal @AdderTest {myParam = 42, myTag = "hello"} {
      %0 = verif.symbolic_value : i42
      %1 = verif.symbolic_value : i42
      %2 = hw.instance "dut" @Adder(a: %0: i42, b: %1: i42) -> (c: i42)
      %3 = comb.add %0, %1 : i42
      %4 = comb.icmp eq %2, %3 : i42
      verif.assert %4 : i1
    }
    ```

    ### Parameters
    The following parameters have a predefined meaning and are interpreted by
    tools such as `circt-test` to guide execution of tests:

    - `ignore`: Indicates whether the test should be ignored and skipped. This
      can be useful for temporarily disabling tests without having to remove
      them from the input. Must be a _boolean_ value.
      ```mlir
      verif.formal @Foo {ignore = true}
      ```

    - `require_runners`: A list of test runners that may be used to execute this
      test. This option may be used to force a test to run using one of a few
      known-good runners, acting like a whitelist. Must be an _array_ of
      _strings_.
      ```mlir
      verif.formal @Foo {require_runners = ["sby", "circt-bmc"]}
      ```

    - `exclude_runners`: A list of test runners that must not be used to execute
      this test. This option may be used to exclude a few known-bad runners from
      executing this test, acting like a blacklist. Must be an _array_ of
      _strings_.
      ```mlir
      verif.formal @Foo {exclude_runners = ["sby", "circt-bmc"]}
      ```
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    DictionaryAttr:$parameters
  );
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name $parameters attr-dict-with-keyword $body";
  let extraClassDeclaration = [{
    static RegionKind getRegionKind(unsigned index) { 
      return RegionKind::Graph;
    }
  }];
}

def SymbolicValueOp : VerifOp<"symbolic_value", [
  ParentOneOf<["verif::FormalOp", "hw::HWModuleOp"]>,
]>{
  let summary = "Create a symbolic value for formal verification";
  let description = [{
    This operation creates a new symbolic value that can be used to formally
    verify designs. Verification tools will try to find concrete assignments for
    symbolic values that violate asserts or make covers true. This value is not
    fixed - the value taken can vary arbitrarily between timesteps.
  }];
  let results = (outs Res<AnyType, "", [MemAlloc]>:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Formal Contracts
//===----------------------------------------------------------------------===//

def ContractOp : VerifOp<"contract", [
  AllTypesMatch<["inputs", "outputs"]>,
  NoRegionArguments,
  NoTerminator,
  RegionKindInterface,
  SingleBlock,
]>{
  let summary = "A formal contract";
  let description = [{
    This operation creates a new formal contract which can be used to locally
    verify a part of the IR and provide simplifying substitutions. Contracts
    contain `verif.require` ops to establish conditions that must hold for a
    piece of IR to work properly, and `verif.ensure` ops to describe the
    properties that the piece of IR must guarantees when the requirements hold.
    Outside of formal verification, operands are simply passed through to the
    results.

    Contracts are checked by extracting them into their own `verif.formal` test
    and replacing `require` with `assume` and `ensure` with `assert`. The
    results of the contract are replaced with the operands of the contract.

    Contracts are used as simplifications for other verification tasks by
    inlining them and replacing `require` with `assert` and `ensure` with
    `assume`. The results of the contract are replaced with symbolic values.

    See the documentation of the Verif dialect for more details.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    ($inputs^ `:` type($inputs))? attr-dict-with-keyword $body
  }];
  let extraClassDeclaration = [{
    static RegionKind getRegionKind(unsigned index) { 
      return RegionKind::Graph;
    }
  }];
}

class RequireLikeOp<string mnemonic> : VerifOp<mnemonic, [
  HasParent<"verif::ContractOp">,
  DeclareOpInterfaceMethods<RequireLike>
]> {
  let arguments = (ins
    LTLAnyPropertyType:$property,
    Optional<I1>:$enable,
    OptionalAttr<StrAttr>:$label
  );
  let assemblyFormat = [{
    $property
    (`if` $enable^)?
    (`label` $label^)?
    attr-dict `:` type($property)
  }];
}

def RequireOp : RequireLikeOp<"require"> {
  let summary = "A precondition of a contract";
  let description = [{
    This operation specifies a condition that is assumed when checking against
    the contract, and asserted when applying the contract as a simplification.

    The `verif.require` op is commonly used to specify the conditions that input
    values into a part of the IR must fulfill in order for the IR to work as
    expected, i.e., as outlined in the contract.
  }];
}

def EnsureOp : RequireLikeOp<"ensure">{
  let summary = "A postcondition of a contract";
  let description = [{
    This operation specifies a condition that is asserted when checking a
    contract, and assumed when applying the contract as a simplification.

    The `verif.ensure` op is commonly used to specify the conditions that output
    values from a part of the IR are guaranteed to fulfill, under the condition
    that all requirements are fulfilled.
  }];
}

#endif // CIRCT_DIALECT_VERIF_VERIFOPS_TD
