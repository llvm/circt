//===- PIRTypes.td - Property IR dialect types -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_PIR_PIRTYPES_TD
#define CIRCT_DIALECT_PIR_PIRTYPES_TD

include "circt/Dialect/PIR/PIRDialect.td"
include "mlir/IR/AttrTypeBase.td"

class PIRTypeDef<string name, string typeMnemonic> : TypeDef<PIRDialect, name> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// Properties
//===----------------------------------------------------------------------===//

def PIRPropertyType : PIRTypeDef<"Simple Property", "property"> {
  let summary = "Simple Property IR Property";
  let description = [{
    Represents a property (in the SVA sense) that is *simple*.
    This means that is only bound to the global clock, i.e. the one whose ticks represent 
    the steps of the model checker used to prove this property. 
    This should not be explicitly bound to any local clock.
    Additionally simple properties must not contain any subsequences that 
    admit empty matches.
  }];
}

def PIRClockedPropertyType : PIRTypeDef<"Clocked Property", "clocked_property"> {
  let summary = "Clocked Property IR Property";
  let description = [{
    Represents a clocked property (in the SVA sense).
    This means that it is bound to an explicit clock. 
    These can be rewritten as simple properties by encoding the clock as an explicit 
    state that depends on the global clock, and having updates for the clocked property 
    wait on the that state holding to be sampled.  
  }];
}

//===----------------------------------------------------------------------===//
// Sequences
//===----------------------------------------------------------------------===//

def PIRSequenceType : PIRTypeDef<"Simple Sequence", "sequence"> {
  let summary = "Simple Property IR Sequence";
  let description = [{
    Represents a sequence (in the SVA sense) that is *simple*.
    This means that is only bound to the global clock, i.e. the one whose ticks represent 
    the steps of the model checker used to prove this property. 
    This should not be explicitly bound to any local clock.
  }];
}

def PIRClockedSequenceType : PIRTypeDef<"Clocked Sequence", "clocked_sequence"> {
  let summary = "Clocked Sequence IR Sequence";
  let description = [{
    Represents a clocked sequence (in the SVA sense).
    This means that it is bound to an explicit clock. 
    These can be rewritten as simple sequences by encoding the clock as an explicit 
    state that depends on the global clock, and having updates for the clocked sequence 
    wait on the that state holding to be sampled.  
  }];
}

//===----------------------------------------------------------------------===//
// Group Types
//===----------------------------------------------------------------------===//

// Accept any sequence, simple or clocked
def PIRAnySequenceType : AnyTypeOf<[PIRSequenceType, PIRClockedSequenceType]>;

// Accept any property, simple or clocked
def PIRAnyPropertyType : AnyTypeOf<[PIRPropertyType, PIRClockedPropertyType]>;

// Accept any clocked type, this includes booleans
def PIRAnyClockedType : AnyTypeOf<[PIRClockedPropertyType, PIRClockedSequenceType, I1]>;

// Accept any simple type, this also includes booleans
def PIRAnySimpleType : AnyTypeOf<[PIRPropertyType, PIRSequenceType, I1]>;

// Accept anything that looks like a valid property
def PIRAnyType : AnyTypeOf<[
  PIRPropertyType, PIRSequenceType, 
  PIRClockedPropertyType, PIRClockedSequenceType,
  I1
]>;

#endif // CIRCT_DIALECT_PIR_PIRTYPES_TD
