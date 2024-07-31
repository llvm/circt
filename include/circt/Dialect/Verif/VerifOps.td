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
    design that have been externalised and appended to the region's
    inputs/outputs (these values are fed from each `circuit` region execution
    to the next, as they represent register state, rather than being
    overwritten with fresh variables like other inputs). The `circuit` region
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

  let arguments = (ins I32Attr:$bound, I32Attr:$num_regs);
  let regions = (region SizedRegion<1>:$init,
                        SizedRegion<1>:$loop,
                        SizedRegion<1>:$circuit);

  let results = (outs I1:$result);

  let assemblyFormat = [{
    `bound` $bound `num_regs` $num_regs attr-dict-with-keyword `init` $init `loop` $loop `circuit`
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
    "verif::InstanceOp"
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
// Verification intent ops
//===----------------------------------------------------------------------===//

def FormalOp : VerifOp<"formal", [
  NoTerminator, RegionKindInterface, HasParent<"mlir::ModuleOp">, 
  IsolatedFromAbove, Symbol
]> {
  let summary = "defines a formal verification test";
  let description = [{
    This operation defines a formal verification test. This should be written 
    by declaring symbolic values that are then connected to a hardware instance. 
    These symbols can then be used in additional assertions that are defined 
    outside of the instantiated `hw.module` but inside of this region. 
    Assertions/Assumptions defined within the instantiated module will also be 
    used for verification. The region can then be converted into btor2, 
    SystemVerilog, or used for verification directly in circt-bmc. The operations 
    in this region are ignored during regular SystemVerilog emission. This allows 
    for verification specific logic to be isolated from the design in a way that 
    is similar to layers.   
      
    The attribute `k` defines the upper bound of cycles to check for this test 
    w.r.t. the implicit clock defined by this operation within its region.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name, APIntAttr:$k);

  let regions = (region SizedRegion<1>:$test_region);

  let assemblyFormat = [{
    $sym_name `(``k` `=` $k `)` attr-dict-with-keyword
    $test_region
  }];

  let extraClassDeclaration = [{
    /// Implement RegionKindInterface.
    static RegionKind getRegionKind(unsigned index) { 
      return RegionKind::Graph;
    }
  }];
}

//===----------------------------------------------------------------------===//
// Formal Verification Ops
//===----------------------------------------------------------------------===//

def SymbolicInputOp : VerifOp<"symbolic_input", [
  ParentOneOf<["verif::FormalOp", "verif::InstanceOp"]>
]>{
  let summary = "declare a symbolic input for formal verification";
  let description = [{
    This operation declares a symbolic input that can then be used to reason 
    about `hw.instance` inputs in a symbolic manner in assertions and 
    assumptions. The result type must be explicitly marked. The resulting value 
    is a new non-deterministic value for every clock cycle of the implicit clock 
    defined by the parent `verif.formal`.  
  }];
  let results = (outs AnyType:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
}

def ConcreteInputOp : VerifOp<"concrete_input", [
  AllTypesMatch<["init", "update", "result"]>, HasParent<"verif::FormalOp">
]> {
  let summary = "declare a concrete input for formal verification";
  let description = [{
    This operation declares a concrete input that can then be used in the 
    reasoning surrounding symbolic inputs, allowing for a form of concolic 
    reasoning to take place in a `verif.formal` block. Concrete inputs are 
    defined by an initial value and an update SSA value. This is equivalent 
    to a register with the parent `verif.formal`'s implicit clock, which is 
    initialized with the init value and obtains a new value every implicit 
    clock tick from the update value.
  }];

  let arguments = (ins AnyType:$init, AnyType:$update);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $init `,` $update attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Formal Contract Ops
//===----------------------------------------------------------------------===//

class ContractLikeOp<string mnemonic, list<Trait> traits = []> 
  : VerifOp<mnemonic, traits # [
    HasParent<"hw::HWModuleOp">, RegionKindInterface, IsolatedFromAbove
  ]> {

  let arguments = (ins Variadic<AnyType>:$inputs);

  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    /// Implement RegionKindInterface.
    static RegionKind getRegionKind(unsigned index) { 
      return RegionKind::Graph;
    }

    /// Retrieves the region block arguments
    BlockArgument getRegionArg(unsigned index) {
      return getBody().front().getArguments()[index];
    }

    /// Retrieves the number of block arguments
    unsigned getNumRegionArgs() {
      return getBody().front().getNumArguments();
    }
  }];

  let hasRegionVerifier = 1;
}


def ContractOp : ContractLikeOp<"contract", [NoTerminator]>{
  let summary = "declare a formal contract";
  let description = [{
    This operation declares a formal contract that is used to create precondition 
    and postconditions on a parent module. These are used as an abstraction to 
    better modularize formal verification such that each module containing a contract 
    is checked exactly once. The contract contains a single block where the block arguments 
    represent the inputs and outputs in the same order as in the module signature.
          
    e.g. 
    ```
    hw.module @Bar(in %foo : i8, out "" : i8, out "1" : i8) { 
      verif.contract (%foo) : (i8) {
        ^bb0(%arg1 : i8, %bar.0 : i8, %bar.1 : i8): 
          %c0_8 = hw.constant 0 : i8 
          %prec = comb.icmp bin ugt %arg1, %c0_8 : i8
          verif.require %prec : i1

          %post = comb.icmp bin ugt %bar.0, %arg1 : i8
          %post1 = comb.icmp bin ult %bar.1, %arg1 : i8
          verif.ensure %post : i1
          verif.ensure %post1 : i1
      }
      /* ... Module definition ... */
    }
    ```

    This later is used to replace any instance of Bar during verification:  
    ```
    %bar.0, %bar.1 = hw.instance "bar" @Bar("foo" : %foo : i8)  -> ("" : i8, "1" : i8)

    /* ... After PrepareForFormal Pass becomes ... */

    %bar.0, %bar.1 = verif.instance (%c42_8) : (i8) -> (i8, i8) {
      ^bb0(%arg1: i8):
        %c0_8 = hw.constant 0 : i8
        %prec = comb.icmp bin ugt %arg1, %c0_8 : i8
        verif.assert %prec : i1

        %bar.0 = verif.symbolic_input : i8
        %bar.1 = verif.symbolic_input : i8
        %post = comb.icmp bin ugt %bar.0, %arg1 : i8
        %post1 = comb.icmp bin ult %bar.1, %arg1 : i8
        verif.assume %post : i1
        verif.assume %post1 : i1
        verif.yield %bar.0, %bar.1 : i8, i8
    }
    
    ```
  }];

  let assemblyFormat = [{
    `(` $inputs `)` attr-dict `:` `(` type($inputs) `)` $body
  }];
}

def InstanceOp : ContractLikeOp<"instance", [
  SingleBlockImplicitTerminator<"verif::YieldOp">
]>{
  let summary = "declare an instance of a formal contract (replace an hw.instance)";
  let description = [{
    This operation declares an instance of an `hw.module` that contains a 
    `verif.contract` definition. The body of this op will be a modified version 
    of the body of the referenced module's contract body, with `verif.require` 
    ops being reoplaced with `verif.assert` and `verif.ensure` being replaced 
    with `verif.assume`. The `verif.result` ops are converted into `verif.symbolic_input` 
    and are yielded as a result of the region to be used in the rest of the module.
  }];

  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    `(` $inputs `)` attr-dict `:` functional-type($inputs, $results) $body
  }];
}

class RequireLikeOp<string mnemonic, list<Trait> traits = [
  HasParent<"verif::ContractOp">, AttrSizedOperandSegments
]> : VerifOp<mnemonic, traits> {
  let arguments = (ins LTLAnyPropertyType:$property,
                      OptionalAttr<ClockEdgeAttr>:$edge, Optional<I1>:$clock,
                      Optional<I1>:$enable,
                      OptionalAttr<StrAttr>:$label);
  let assemblyFormat = [{
    $property (`if` $enable^)? (`,` $edge^ $clock)? 
    (`label` $label^)? attr-dict `:` type($property)
  }];
}

def RequireOp : RequireLikeOp<"require"> {
  let summary = "define a precondition for the given contract";
  let description = [{
    This operation defines a precondition for the parent contract. 
    Preconditions are assumed during the verification of a module and 
    asserted during the verification of an instance of a module.
  }];
}


def EnsureOp : RequireLikeOp<"ensure">{
  let summary = "define a postcondition for the given contract";
  let description = [{
    This operation defines a postcondition for the parent contract. 
    Postconditions are asserted during the verification of a module and 
    assumed during the verification of an instance of a module.
  }];
}

#endif // CIRCT_DIALECT_VERIF_VERIFOPS_TD
