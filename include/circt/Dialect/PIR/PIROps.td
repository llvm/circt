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
include "mlir/IR/EnumAttr.td"
include "mlir/IR/PatternBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class PIROp<string mnemonic, list<Trait> traits = []> :
  Op<PIRDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Assertions
//===----------------------------------------------------------------------===//

class AssertPropertyLikeOp<string mnemonic, list<Trait> traits = []> :
  PIROp<mnemonic, traits> {
  let arguments = (
    ins PIRClockedPropertyType:$property, 
    Optional<I1>:$disable,
    Optional<I1>:$enable
  );
  let assemblyFormat = [{
    $property (`disable` `iff` $disable^)? (`on` $enable^)? attr-dict `:` type($property)
  }];
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

def RestrictPropertyOp : AssertPropertyLikeOp<"restric_property"> {
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
// Clocking
//===----------------------------------------------------------------------===//



//===----------------------------------------------------------------------===//
// Sequences
//===----------------------------------------------------------------------===//




//===----------------------------------------------------------------------===//
// Properties
//===----------------------------------------------------------------------===//


#endif // CIRCT_DIALECT_PIR_PIROPS_TD
