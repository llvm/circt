//===- PIROps.td - Property IR dialect operations ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_PIR_PIROPS_TD
#define CIRCT_DIALECT_PIR_PIROPS_TD

include "circt/Dialect/HW/HWTypes.td"
include "circt/Dialect/PIR/PIRDialect.td"
include "circt/Dialect/PIR/PIRTypes.td"
include "circt/Dialect/PIR/PIREnums.td"
include "circt/Dialect/Seq/SeqTypes.td"
include "circt/Dialect/HW/HWEnums.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/PatternBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"

class PIROp<string mnemonic, list<Trait> traits = []> :
  Op<PIRDialect, mnemonic, traits>;

class PIRInnerOp<string mnemonic, list<Trait> traits = [
  HasParent<"pir::CellOp">
]> : PIROp<mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Top-level operations
//===----------------------------------------------------------------------===//

def CellOp : PIROp<"cell", [
  IsolatedFromAbove,
  NoRegionArguments,
  NoTerminator,
  RegionKindInterface,
  SingleBlock,
  SymbolName, 
  SymbolVisibility, 
  Symbol
]> {
  let summary = "PIR Wrapper Cell for Properties and Sequences.";
  let description = [{
    Wrapper cell which encapsulates all of the logic related to a specific 
    property or sequence. This allows PIR properties to be both well contained 
    and exist within any other dialect. When interoperating with Yosys, the 
    contents of a cell will be the only information shared with Yosys infrastructure.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility,
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

def InputOp : PIRInnerOp<"input", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  SameVariadicResultSize
]> {
  let summary = "Defines the inputs to a PIR Cell.";
  let description = [{
    Defines the inputs to a PIR Cell as a single variadic symbolic value.
    This can of course also be used to model non-determinism or `ANY_SEQ` 
    macros from YOSYS produced SystemVerilog into CIRCT, e.g.
    ```mlir
    pir.cell @prop1 {
      %a, %b, %c = pir.input : i1, i8, i8
    }
    ```
    In the example above, the inputs `a`, `b`, and `c` can then be used to 
    reference signals from the surrounding design in a manner that allows 
    the logic of the property to remain isolated from the logic of the design 
    under verification (DUV).
  }];

  let arguments = (ins StrArrayAttr:$names);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "`` custom<ImplicitSSANames>($names) attr-dict `:` type($results)";
}

//===----------------------------------------------------------------------===//
// Assertions
//===----------------------------------------------------------------------===//

class AssertPropertyLikeOp<string mnemonic, list<Trait> traits = [
  AttrSizedOperandSegments,
]> : PIRInnerOp<mnemonic, traits> {
  let arguments = (ins 
    PIRClockedPropertyType:$property,
    OptionalAttr<ScopeAttr>:$scope,
    Optional<I1>:$disable,
    Optional<I1>:$enable
  );
  let assemblyFormat = [{
    $property ($scope^)? (`disable` `iff` $disable^)? (`on` $enable^)? attr-dict `:` type($property)
  }];

  let hasCanonicalizer = true;
}

def AssertPropertyOp : AssertPropertyLikeOp<"assert_property"> {
  let summary = "Assert that a Property IR property holds.";
  let description = [{
    Concurrent assertion that checks if the current property holds.
    This only accepts clocked properties, thus all other types must 
    be explicitly casted to a clocked property, using an appropriate type 
    conversion operation, before they can be asserted.
  }];
}

def AssumePropertyOp : AssertPropertyLikeOp<"assume_property"> {
  let summary = "Assumes that a Property IR property holds.";
  let description = [{
    Concurrent assumption that checks if the current property holds.
    This only accepts clocked properties, thus all other types must 
    be explicitly casted to a clocked property, using an appropriate type 
    conversion operation, before they can be assumed. 
    Concurrent assumptions are treated alongside assertions without directly 
    impacting their timing. As such if an assumption turns out to not hold, then 
    ongoing assertions that depend on it will be disabled as the assumed property 
    terminates.
  }];
}

def RestrictOp : AssertPropertyLikeOp<"restrict"> {
  let summary = "Restricts the input space checked for a given property.";
  let description = [{
    Similarly to Assumptions, restrictions allow for a property's input space to be 
    constrained. They differ mainly in simulation semantics, where an assumption must 
    be checked before it is applied, a restriction is just applied regardless of its 
    validity. Restrict is identical to assume in the context of formal verification.
  }];
}

def CoverPropertyOp : AssertPropertyLikeOp<"cover_property"> {
  let summary = "Cover-point that tracks how often a property holds.";
  let description = [{
    Tracks how often a concurrent property holds. Will yeald a coverage report 
    in the end. Mostly makes sense in the context of finite traces, e.g. simulation.
  }];
}

//===----------------------------------------------------------------------===//
// Type Conversions
//===----------------------------------------------------------------------===//

class ConversionOp<string mnemonic, list<Trait> traits = [Pure]> :
  PIRInnerOp<mnemonic, traits> {
    let assemblyFormat = [{
      $input attr-dict
    }];
}

def BoolToClockedSeqOp : ConversionOp<"bool_to_clocked_seq"> {
  let summary = "Convert a Boolean expression to a clocked sequence of length 1.";
  let description = [{
    Converts a Boolean expression to a clocked sequence of length 1. 
    In SystemVerilog, there exists no equivalent operator because this happens implicitly 
    by using a Boolean expression in a sequence context. This uses the global clock.
  }];

  let arguments = (
    ins I1:$input
  );
  let results = (
    outs PIRClockedSequenceType:$output
  );
}

def SeqToClockedSeqOp : ConversionOp<"seq_to_clocked_seq"> {
  let summary = "Convert a simple sequence into a clocked sequence.";
  let description = [{
    Convert a simple sequence into a clocked sequence that uses the global clock. 
    Then it can be used as a parameter for assertions. Note that it is not possible 
    to change the clock that is used by the resulting clocked sequence.
  }];

  let arguments = (
    ins PIRSequenceType:$input
  );
  let results = (
    outs PIRClockedSequenceType:$output
  );
}

def BoolToClockedPropOp : ConversionOp<"bool_to_clocked_prop"> {
  let summary = "Convert a Boolean expression to a clocked property of length 1.";
  let description = [{
    Converts a Boolean expression directly to a sequence property.
    Note that sequence properties must not admit empty matches.
  }];

  let arguments = (
    ins I1:$input
  );
  let results = (
    outs PIRClockedPropertyType:$output
  );
}

def ClockedSeqToClockedPropOp : ConversionOp<"clocked_seq_to_clocked_prop"> {
  let summary = "Converts a clocked sequence to sequence property.";
  let description = [{
    Converts a clocked sequence to sequence property using the same clock. 
  }];

  let arguments = (
    ins PIRClockedSequenceType:$input
  );
  let results = (
    outs PIRClockedPropertyType:$output
  );

  let hasCanonicalizeMethod = 1;
}


//===----------------------------------------------------------------------===//
// Clocking
//===----------------------------------------------------------------------===//

def ClockedOp : PIRInnerOp<"clocked", [
  Pure, 
  InferTypeOpInterface, 
  DeclareOpInterfaceMethods<InferTypeOpInterface>
]> {
  let summary = "Specify the clock of a clocked operation.";
  let description = [{
    Specifies the clock for a given clocked sequence or clocked property.
    Note that these must be converted into a "clocked" type before being given 
    a specific clock.
  }];

  let arguments = (ins
    AnyClockType: $clock, 
    PIRAnyClockedType:$input
  );

  let results = (outs
    PIRAnyClockedType:$output
  );

  let assemblyFormat = [{
    $clock `,` $input attr-dict `:` qualified(type($clock)) `,` qualified(type($input))
  }];
}

def GlobalClockSampleOp : PIRInnerOp<"gclk_sample", [Pure]> {
  let summary = "Samples the input with respect to the global clock.";
  let description = [{
    Samples a given input signal with respect to the global clock on a given 
    clock edge. This is useful when encoding other clocks through their relation to the 
    global clock.

    Additionally, the two arguments of this operation are used to represent 
    clocks that might have undefined traces, i.e. moments where the clocks 
    evaluate to either `x` or `z`. 
    These arguments are used as follows: 
    
      operand[0] | operand[1] | result
      ----------------------------------
            0    |     1      |    0
            1    |     1      |    1
            0    |     0      |    z
            1    |     0      |    x

    Operand[0] is referred to as "clock_value" as it carries the value of the 
    clock, while operand[1] is referred to as "clock_defined" as it specifies 
    whether the clock is currently in an undefined state.

    Clock_defined can be omitted, in which case it is interpreted as a constant 1.
  }];

  let arguments = (ins
    EventControlAttr:$edge,
    I1:$clockValue,
    Optional<I1>:$clockDefined
  );

  let results = (outs
    I1:$result
  );

  let assemblyFormat = [{
    $edge $clockValue (`,` $clockDefined^)? attr-dict
  }];
}

def GlobalClockFutureOp : PIRInnerOp <"gclk_future", [Pure]> {
  let summary = "Samples the value of the input at the next gobal time step.";
  let description = [{
    Samples the value of the input at the next gobal time step.
    This allows us e.g. to model any clock in terms of the global clock.
    Note that the three other global clock sampling methods can also be modeled using 
    future sampling as follows:
    ```mlir
    %res = pir.glck_sample posedge %clk_v, %clk_d : i1 
    
    // is equivalent to:

    %0 = comb.and bin %clk_v, %clk_d : i1
    %true = hw.constant true
    %1 = comb.xor bin %0, %true : i1
    %2 = pir.gclk_future %0 : i1
    %res = comb.and bin %2, %1 : i1

    // -----

    %res = pir.glck_sample negedge %clk_v, %clk_d : i1 
    
    // is equivalent to:

    %true = hw.constant true
    %n_clk_v = comb.xor bin %clk_v, %true : i1
    %0 = comb.and bin %n_clk_v, %clk_d : i1
    %1 = comb.xor bin %0, %true : i1
    %2 = pir.gclk_future %0 : i1
    %res = comb.and bin %2, %1 : i1

    // -----

    %res = pir.glck_sample edge %clk_v, %clk_d : i1 
    
    // is equivalent to:

    %f0 = gclk_future %clk_v : i1
    %f1 = gclk_future %clk_d : i1
    %0 = comb.xor bin %clk_v, %f0 : i1
    %1 = comb.xow bin %clk_d, %1 : i1
    %res = comb.or bin %0, %1 : i1
    ```  
  }];

  let arguments = (ins
    I1:$input
  );

  let results = (outs
    I1:$result
  );

  let assemblyFormat = [{
    $input attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Sequences
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// Properties
//===----------------------------------------------------------------------===//


#endif // CIRCT_DIALECT_PIR_PIROPS_TD
