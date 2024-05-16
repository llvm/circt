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
    The `ltl.clocked_sequence` type represents a sequence of linear temporal logic, 
    associated to a specific clock for example, *"A is true two cycles after B is true"*.

    Note that this type explicitly identifies a *clocked sequence*, i.e., a boolean
    value (`i1`), or the result of a delay or concatenation, associated to a clock. 
    Operations that accept a sequence as an operand will use the `AnySequence` constraint, 
    which also accepts `i1`.
  }];
}

def LTLPropertyType : LTLTypeDef<"Property", "property"> {
  let summary = "LTL property type";
  let description = [{
    The `ltl.property` type represents a verifiable property built from linear
    temporal logic sequences and quantifiers, for example, *"if you see sequence
    A, eventually you will see sequence B"*.

    Note that this type explicitly identifies a *property*. However, a boolean
    value (`i1`) or a sequence (`ltl.sequence`) is also a valid property.
    Operations that accept a property as an operand will use the `AnyProperty`
    constraint, which also accepts `ltl.sequence` and `i1`.
  }];
}

def LTLDisabledPropertyType : LTLTypeDef<"DisabledProperty", "disabled_property"> {
  let summary = "LTL disabled property type";
  let description = [{
    The `ltl.disabled_property` type represents a verifiable property built from linear
    temporal logic sequences and quantifiers that has explicitly been given a disable condition, 
    for example, *"if you see sequence A, eventually you will see sequence B unless D holds"*.

    Note that this type explicitly identifies a *disabled property*, i.e. any `i1`, 
    `ltl.sequence` or `ltl.property` that was disabled using `ltl.disable`.
  }];
}

def LTLClockedPropertyType : LTLTypeDef<"ClockedProperty", "clocked_property"> {
  let summary = "LTL clocked property type";
  let description = [{
    The `ltl.clocked_property` type represents a verifiable property built from linear
    temporal logic sequences and quantifiers that has explicitly been associated to a clock signal.

    Note that this type explicitly identifies a *clocked property*, i.e. any `i1`, 
    `ltl.sequence` or `ltl.property` that was tied to a clock using `ltl.clock`.
  }];
}

def LTLClockedDisabledPropertyType : LTLTypeDef<"ClockedDisabledProperty", "clocked_disabled_property"> {
  let summary = "LTL clocked and disabled property type";
  let description = [{
    The `ltl.clocked_disabled_property` type represents a verifiable property built from linear
    temporal logic sequences and quantifiers that has explicitly been associated to a clock signal and 
    and has a disable condition.

    Note that this type explicitly identifies a *clocked property*, i.e. any `i1`, 
    `ltl.sequence` or `ltl.property` that was tied to a clock using `ltl.clock` and 
    given a disable condition using `ltl.disable`.
  }];
}

def LTLAnySequenceType : AnyTypeOf<[I1, LTLSequenceType, LTLClockedSequenceType]>;
def LTLAnyPropertyType : AnyTypeOf<[I1, LTLSequenceType, LTLClockedSequenceType, LTLPropertyType, 
    LTLDisabledPropertyType, LTLClockedPropertyType, LTLClockedDisabledPropertyType]>;
def LTLAnyPurePropertyType : AnyTypeOf<[I1, LTLSequenceType, LTLPropertyType]>;
def LTLAnyNonDisabledPropertyType : AnyTypeOf<[I1, LTLSequenceType, LTLClockedSequenceType, 
    LTLPropertyType, LTLClockedPropertyType]>;
def LTLAnyClockedType : AnyTypeOf<[LTLClockedSequenceType, LTLClockedPropertyType, LTLClockedDisabledPropertyType]>;

#endif // CIRCT_DIALECT_LTL_LTLTYPES_TD
