//===- LTLTypes.td - LTL dialect types ---------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_LTL_LTLTYPES_TD
#define CIRCT_DIALECT_LTL_LTLTYPES_TD

include "circt/Dialect/LTL/LTLDialect.td"
include "mlir/IR/AttrTypeBase.td"

class LTLTypeDef<string name, string typeMnemonic> : TypeDef<LTLDialect, name> {
  let mnemonic = typeMnemonic;
}

def LTLSequenceType : LTLTypeDef<"Sequence", "sequence"> {
  let summary = "LTL sequence type";
  let description = [{
    The `ltl.sequence` type represents a sequence of linear temporal logic, for
    example, *"A is true two cycles after B is true"*.

    Note that this type explicitly identifies a *sequence*. However, a boolean
    value (`i1`) is also a valid sequence. Operations that accept a sequence as
    an operand will use the `AnySequence` constraint, which also accepts `i1`.
  }];
}

def LTLClockedSequenceType : LTLTypeDef<"ClockedSequence", "clocked_sequence"> {
  let summary = "LTL clocked sequence type";
  let description = [{
    The `ltl.clocked_sequence` type is the fundamental type for representing
    temporal sequences in linear temporal logic, for example, *"on the rising
    edge of clk, A is true two cycles after B is true"*.

    This type intrinsically carries its clocking context, explicitly binding a
    temporal sequence of events to a specific clock. Operations like ltl.concat,
    ltl.and, and ltl.or operate on `ltl.clocked_sequence` operands, ensuring
    all sequence operands share the same clock. This makes clock domains
    explicit and type-checked, preventing errors and clarifying clock domains
    for analysis passes.

    Boolean inputs (`i1`) are implicitly lifted to a zero-length clocked
    sequence. Operations that accept a clocked sequence as an operand will use
    the `AnyClockedSequence` constraint, which also accepts `i1`.
  }];
}

def LTLPropertyType : LTLTypeDef<"Property", "property"> {
  let summary = "LTL property type";
  let description = [{
    The `ltl.property` type represents a clock-agnostic, verifiable property
    built from explicitly clocked sequences and quantifiers, for example, *"if
    you see sequence A, eventually you will see sequence B"*.

    This type is fundamentally clock-independent - it represents a quantified
    statement about explicitly clocked sequences rather than being a clocked
    entity itself. The "lifting" from clocked sequences to clock-agnostic
    properties occurs via property operators like `ltl.implication`,
    `ltl.always`, and `ltl.eventually`.

    Properties can describe relationships between temporal patterns that may
    exist in different clock domains, enabling cross-domain verification.
  }];
}

def LTLAnySequenceType : AnyTypeOf<[I1, LTLSequenceType]>;
def LTLAnyPropertyType : AnyTypeOf<[I1, LTLSequenceType, LTLPropertyType]>;
def LTLSequenceOrPropertyType : AnyTypeOf<[LTLSequenceType, LTLPropertyType]>;

#endif // CIRCT_DIALECT_LTL_LTLTYPES_TD
