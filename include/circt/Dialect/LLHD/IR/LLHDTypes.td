//===- LLHDTypesImpl.td - LLHD data type definitions -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Basic data type implementations for the LLHD dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_LLHD_IR_LLHDTYPES_TD
#define CIRCT_DIALECT_LLHD_IR_LLHDTYPES_TD

include "circt/Dialect/LLHD/IR/LLHDDialect.td"
include "circt/Dialect/HW/HWTypes.td"
include "mlir/IR/AttrTypeBase.td"

// Base class for other typedefs. Provides dialact-specific defaults.
class LLHDType<string name> : TypeDef<LLHDDialect, name> { }

//===----------------------------------------------------------------------===//
// Type declarations
//===----------------------------------------------------------------------===//

def LLHDTimeType : LLHDType<"Time"> {
  let summary = "time type";
  let description = [{
    Represents a simulation time value as a combination of a real time value in
    seconds (or any smaller SI time unit), a delta value representing
    infinitesimal time steps, and an epsilon value representing an absolute time
    slot within a delta step (used to model SystemVerilog scheduling regions).
  }];
  let mnemonic = "time";
}

def RefType : LLHDType<"Ref"> {
  let summary = "a reference to a signal or variable";
  let description = [{
    Represents a reference to a value. Signals return a reference to the
    underlying value, which allows them to be probed or driven.
  }];
  let mnemonic = "ref";
  let parameters = (ins "Type":$nestedType);
  let assemblyFormat = "`<` $nestedType `>`";
  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$nestedType), [{
      return $_get(nestedType.getContext(), nestedType);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// Type Constraints
//===----------------------------------------------------------------------===//


/// A ref type with a specific nested type.
class RefTypeOf<Type type> : ConfinedType<RefType,
  [SubstLeaves<"$_self", "cast<llhd::RefType>($_self).getNestedType()",
               type.predicate>],
  "ref of " # type.summary, "llhd::RefType"
> {
  Type nestedType = type;
}

class TypeMatchesRefNestedType<string type, string refType> : TypesMatchWith<
  type # " type must match " # refType # " ref element type",
  type, refType, "RefType::get($_self)"
>;

class RefNestedTypeMatchesType<string refType, string type> : TypesMatchWith<
  refType # " ref element type must match " # type # " type",
  refType, type, "cast<RefType>($_self).getNestedType()"
>;

//===----------------------------------------------------------------------===//
// Attribute declarations
//===----------------------------------------------------------------------===//

def LLHDTimeAttr : AttrDef<LLHDDialect, "Time", []> {
  let summary = "time attribute";
  let description = [{
    Represents a value of the LLHD time type.

    Example: `#llhd.time<0ns, 1d, 0e>`
  }];

  let mnemonic = "time";
  let storageType= [{ llhd::TimeAttr }];
  let genVerifyDecl = 1;

  let hasCustomAssemblyFormat = 1;

  let parameters = (ins AttributeSelfTypeParameter<"", "llhd::TimeType">:$type,
                        "uint64_t":$time,
                        StringRefParameter<"SI time unit">:$timeUnit,
                        "unsigned":$delta,
                        "unsigned":$epsilon);

  let skipDefaultBuilders = 1;
  let builders = [ AttrBuilder<(ins "uint64_t":$time,
                                    "llvm::StringRef":$timeUnit,
                                    "unsigned":$delta,
                                    "unsigned":$epsilon), [{
    return $_get($_ctxt, llhd::TimeType::get($_ctxt),
      time, timeUnit, delta, epsilon);
  }]>];
}

#endif // CIRCT_DIALECT_LLHD_IR_LLHDTYPES_TD
