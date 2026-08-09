//===- LTLOps.td - LTL dialect operations ------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_LTL_LTLOPS_TD
#define CIRCT_DIALECT_LTL_LTLOPS_TD

include "circt/Dialect/HW/HWTypes.td"
include "circt/Dialect/LTL/LTLDialect.td"
include "circt/Dialect/LTL/LTLTypes.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/PatternBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class LTLOp<string mnemonic, list<Trait> traits = []> :
  Op<LTLDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Generic Operations
//===----------------------------------------------------------------------===//

class AssocLTLOp<string mnemonic, list<Trait> traits = []> :
  LTLOp<mnemonic, traits # [Pure, Commutative, InferTypeOpInterface,
                            DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Variadic<LTLAnyPropertyType>:$inputs);
  let results = (outs LTLAnyPropertyType:$result);
  let assemblyFormat = [{
    $inputs attr-dict `:` type($inputs)
  }];
}

def AndOp : AssocLTLOp<"and"> {
  let summary = "A conjunction of booleans, sequences, or properties.";
  let description = [{
    If any of the `$inputs` is of type `!ltl.property`, the result of the op is
    an `!ltl.property`. Otherwise it is an `!ltl.sequence`.
  }];
  let hasCanonicalizeMethod = 1;
}

def OrOp : AssocLTLOp<"or"> {
  let summary = "A disjunction of booleans, sequences, or properties.";
  let description = [{
    If any of the `$inputs` is of type `!ltl.property`, the result of the op is
    an `!ltl.property`. Otherwise it is an `!ltl.sequence`.
  }];
  let hasCanonicalizeMethod = 1;
}

def IntersectOp : AssocLTLOp<"intersect"> {
  let summary = "The intersection of booleans sequences or properties.";
  let description = [{
    The intersection of two properties. This checks that both properties both hold
    and have the same start and end times. This differs from `ltl.and` which doesn't
    consider the timings of each operand, only their results.
  }];
  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// Clocking
//===----------------------------------------------------------------------===//

// Edge behavior enum for always block.  See SV Spec 9.4.2.

/// AtPosEdge triggers on a rise from 0 to 1/X/Z, or X/Z to 1.
def AtPosEdge: I32EnumAttrCase<"Pos", 0, "posedge">;
/// AtNegEdge triggers on a drop from 1 to 0/X/Z, or X/Z to 0.
def AtNegEdge: I32EnumAttrCase<"Neg", 1, "negedge">;
/// AtEdge is syntactic sugar for AtPosEdge or AtNegEdge.
def AtEdge   : I32EnumAttrCase<"Both", 2, "edge">;

def ClockEdgeAttr : I32EnumAttr<"ClockEdge", "clock edge",
                                [AtPosEdge, AtNegEdge, AtEdge]> {
  let cppNamespace = "circt::ltl";
}

def ClockedAtomOp : LTLOp<"clocked_atom", [Pure]> {
  let arguments = (ins I1:$input, ClockEdgeAttr:$edge, I1:$clock);
  let results = (outs LTLSequenceType:$result);
  let assemblyFormat = [{
    $input `,` $edge $clock attr-dict `:` type($input)
  }];

  let summary = "Convert a boolean into a sequence on an explicit clock.";
  let description = [{
    Converts the boolean `$input` into a one-cycle sequence which matches when
    `$input` is true on the specified `$edge` of `$clock`.

    This operation records the clocking event associated with a single boolean
    atom and returns an `!ltl.sequence`.

    For example:

    - `ltl.clocked_atom %a, posedge %clk : i1` matches `%a` on the positive
      edge of `%clk`.
  }];
}

//===----------------------------------------------------------------------===//
// Sequences
//===----------------------------------------------------------------------===//

def ClockedDelayOp : LTLOp<"clocked_delay", [Pure]> {
  let arguments = (ins
    LTLAnySequenceType:$input,
    ClockEdgeAttr:$edge,
    I1:$clock,
    I64Attr:$delay,
    OptionalAttr<I64Attr>:$length);
  let results = (outs LTLSequenceType:$result);
  let assemblyFormat = [{
    $input `,` $edge $clock `,` $delay (`,` $length^)? attr-dict `:` type($input)
  }];
  let hasCanonicalizer = 1;

  let summary = "Delay a sequence by a number of cycles on an explicit clock.";
  let description = [{
    Delays the `$input` sequence by the number of cycles specified by `$delay`
    on the specified `$clock` and `$edge`.

    The optional `$length` specifies during how many cycles after the initial
    delay the sequence can match. Omitting `$length` indicates an unbounded but
    finite delay.

    For example:

    - `ltl.clocked_delay %seq, posedge %clk, 2, 0` delays `%seq` by
      exactly 2 cycles on the positive edge of `%clk`.
    - `ltl.clocked_delay %seq, negedge %clk, 2, 2` delays `%seq` by 2,
      3, or 4 cycles on the negative edge of `%clk`.
  }];
}

def ClockedPastOp : LTLOp<"clocked_past", [TypesMatchWith<"input and result types must match",
    "input", "result", "$_self">,
    Pure]> {
  let arguments = (ins
    HWIntegerType:$input,
    I64Attr:$delay,
    I1:$clk);
  let results = (outs HWIntegerType:$result);
  let assemblyFormat = [{
    $input `,` $delay `clk` $clk attr-dict `:` type($input)
  }];

  let summary = "Observe a value $delay cycles earlier on an explicit clock.";
  let description = [{
      Observes the input `$delay` cycles before the current cycle, sampled by
      the `$clk` operand.
  }];
}

def SampledOp : LTLOp<"sampled", [TypesMatchWith<"expression and sampledValue types must match",
    "expression", "sampledValue", "$_self">,
    Pure]> {
  let summary = "`$sampled` system function to sample a value";
  let description = [{
    Sample a value using System Verilog sampling semantics (see IEEE 1800–2023 $16.5.1 
    "Sampling" for more information).

    A use of `$sampled` is to safely read the value of a net/variable in a
    concurrent assertion action block such that the value will be the same as
    the value used when the assertion is triggered.  See IEEE 1800–2023 $16.9.3 "Sampled 
    value functions" for more information.
  }];
  let arguments = (ins AnyType:$expression);
  let results = (outs AnyType:$sampledValue);
  let assemblyFormat = "$expression attr-dict `:` qualified(type($expression))";
}

def ConcatOp : LTLOp<"concat", [Pure]> {
  let arguments = (ins Variadic<LTLAnySequenceType>:$inputs);
  let results = (outs LTLSequenceType:$result);
  let assemblyFormat = [{
    $inputs attr-dict `:` type($inputs)
  }];
  let hasFolder = 1;
  let hasCanonicalizer = 1;

  let summary = "Concatenate sequences into a longer sequence.";
  let description = [{
    Concatenates all of the `$inputs` sequences one after another into one
    longer sequence. The sequences are arranged such that the end time of the
    previous sequences coincides with the start time of the next sequence. This
    means there is no implicit cycle of delay between the concatenated
    sequences, which may be counterintuitive.

    If a sequence should follow in the cycle after another sequence finishes,
    that cycle of delay needs to be explicit. For example, *"u followed by v in
    next cycle"* (`u ##1 v` in SVA) is represented as
    `concat(u, clocked_delay(v, posedge clk, 1, 0))`:
    ```
    %0 = ltl.clocked_delay %v, posedge %clk, 1, 0 : i1
    ltl.concat %u, %0 : !ltl.sequence, !ltl.sequence
    ```
    The resulting sequence checks for `u` in the first cycle and `v` in the
    second, `[u, v]` in short.

    Without this explicit delay, the previous sequence's end overlaps with the
    next sequence's start. For example, consider the two sequences `u = a ##1 b`
    and `v = c ##1 d`, which check for `a` and `c` in the first, and `b` and `d`
    in the second cycle. When these two sequences are concatenated,
    `concat(u, v)`, the end time of the first sequence coincides with the start
    time of the second. As a result, the check for `b` at the end of the first
    sequence will coincide with the check for `c` at the start of the second
    sequence: `concat(u, v) = a ##1 (b && c) ##1 d`. The resulting sequence
    checks for `a` in the first cycle, `b` and `c` in the second, and `d` in the
    third, `[a, (b && c), d]` in short.

    By making the delay between concatenated sequences explicit, the `concat`
    operation behaves nicely in the presence of zero-length sequences. An empty,
    zero-length sequence in a concatenation behaves as if the sequence wasn't
    present at all. Compare this to SVAs which struggle with empty sequences.
    For example, `x ##1 y ##1 z` would become `x ##2 z` if `y` was empty.
    Similarly, expressing zero or more repetitions of a sequence, `w ##[*]`, is
    challenging in SVA since concatenation always implies a cycle of delay, but
    trivial if the delay is made explicit. This is related to the handling of
    empty rules in a parser's grammar.

    Note that concatenating two boolean values *a* and *b* is equivalent to
    computing the logical AND of them. Booleans are sequences that check if the
    boolean is true in the current cycle, which means that the sequence starts
    and ends in the same cycle. Since concatenation aligns the sequences such
    that end time of *a* and start time of *b* coincide, the resulting sequence
    checks if *a* and *b* both are true in the current cycle, which is an AND
    operation.
  }];
}

def ClockedRepeatOp : LTLOp<"clocked_repeat", [Pure]> {
  let arguments = (ins LTLAnySequenceType:$input, ClockEdgeAttr:$edge, I1:$clock,
                       I64Attr:$base, OptionalAttr<I64Attr>:$more);
  let results = (outs LTLSequenceType:$result);
  let assemblyFormat = [{
    $input `,` $edge $clock `,` $base (`,` $more^)? attr-dict `:` type($input)
  }];
  let summary = "Repeat a sequence on an explicit clock.";
  let description = [{
    Repeats `$input` at least `$base` times and at most `$base + $more` times,
    sampled at the explicit `$edge` of `$clock`. Omitting `$more` indicates an
    unbounded but finite repetition.
  }];
}

def ClockedGoToRepeatOp : LTLOp<"clocked_goto_repeat", [Pure]> {
  let arguments = (ins LTLAnySequenceType:$input, ClockEdgeAttr:$edge, I1:$clock,
                       I64Attr:$base, I64Attr:$more);
  let results = (outs LTLSequenceType:$result);
  let assemblyFormat = [{
    $input `,` $edge $clock `,` $base `,` $more attr-dict `:` type($input)
  }];
  let summary = "Clocked `goto`-style non-consecutive repetition.";
  let description = [{
    Non-consecutively repeats `$input` between `$base` and `$base + $more`
    times, sampled at the explicit `$edge` of `$clock`. The final evaluation
    in the sequence must match. The number of evaluations between matches is
    unbounded.
  }];
}

def ClockedNonConsecutiveRepeatOp : LTLOp<"clocked_non_consecutive_repeat", [Pure]> {
  let arguments = (ins LTLAnySequenceType:$input, ClockEdgeAttr:$edge, I1:$clock,
                       I64Attr:$base, I64Attr:$more);
  let results = (outs LTLSequenceType:$result);
  let assemblyFormat = [{
    $input `,` $edge $clock `,` $base `,` $more attr-dict `:` type($input)
  }];
  let summary = "Clocked non-consecutive repetition.";
  let description = [{
    Non-consecutively repeats `$input` between `$base` and `$base + $more`
    times, sampled at `$edge` of `$clock`. The final evaluation in the sequence
    does not have to match. The number of evaluations between matches is
    unbounded.
  }];
}


//===----------------------------------------------------------------------===//
// Properties
//===----------------------------------------------------------------------===//

def BooleanConstantOp : LTLOp<"boolean_constant", [Pure, ConstantLike]> {
  let summary = "A constant boolean property value.";
  let description = [{
    Represents a constant boolean value as an LTL property. This operation
    takes a boolean attribute and produces an `!ltl.property` result. This is
    useful for representing constant property values in canonicalization patterns.

    Example:
    ```mlir
    %prop = ltl.boolean_constant true
    ```
  }];

  let arguments = (ins BoolAttr:$value);
  let results = (outs LTLPropertyType:$result);
  let assemblyFormat = [{
    $value attr-dict
  }];

  let hasFolder = 1;
}

def NotOp : LTLOp<"not", [Pure]> {
  let arguments = (ins LTLAnyPropertyType:$input);
  let results = (outs LTLPropertyType:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];

  let summary = "A negation of a property.";
  let description = [{
    Negates the `$input` property. The resulting property evaluates to true if
    `$input` evaluates to false, and it evaluates to false if `$input` evaluates
    to true.
  }];
}

def ImplicationOp : LTLOp<"implication", [Pure]> {
  let arguments = (ins LTLAnySequenceType:$antecedent,
                       LTLAnyPropertyType:$consequent);
  let results = (outs LTLPropertyType:$result);
  let assemblyFormat = [{
    operands attr-dict `:` type(operands)
  }];

  let hasFolder = 1;

  let summary = "Only check a property after a sequence matched.";
  let description = [{
    Preconditions the checking of the `$consequent` property on the
    `$antecedent` sequence. In a nutshell, if the `$antecedent` sequence matches
    at a given point in time, the `$consequent` property is checked starting at
    the point in time at which the matched sequence ends. The result property of
    the `ltl.implication` holds if the `$consequent` holds. Conversely, if the
    `$antecedent` does *not* match at a given point in time, the result property
    trivially holds. This is conceptually identical to the implication operator
    →, but with additional temporal semantics.
  }];
}

def ClockedUntilOp : LTLOp<"clocked_until", [Pure]> {
  let arguments = (ins LTLAnyPropertyType:$input, ClockEdgeAttr:$edge, I1:$clock,
                       LTLAnyPropertyType:$condition);
  let results = (outs LTLPropertyType:$result);
  let assemblyFormat = [{
    $input `,` $edge $clock `,` $condition attr-dict `:`
    type($input) `,` type($condition)
  }];

  let summary = "Clocked weak-until property.";
  let description = [{
    Checks that `$input` holds until `$condition` holds once, sampled at the
    explicit `$edge` of `$clock`. This operator is weak and non-overlapping.
  }];
}

def ClockedEventuallyOp : LTLOp<"clocked_eventually", [Pure]> {
  let arguments = (ins LTLAnyPropertyType:$input, ClockEdgeAttr:$edge, I1:$clock);
  let results = (outs LTLPropertyType:$result);
  let assemblyFormat = [{
    $input `,` $edge $clock attr-dict `:` type($input)
  }];
  let summary = "Clocked strong-eventually property.";
  let description = [{
    Checks that `$input` holds after a finite number of cycles, sampled at the
    explicit `$edge` of `$clock`. This operator is strong.
  }];
}

#endif // CIRCT_DIALECT_LTL_LTLOPS_TD
