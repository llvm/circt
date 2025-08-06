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

def LLHDPtrType : LLHDType<"Ptr"> {
  let summary = "pointer type";
  let description = [{
    Represents a pointer to a memory location holding a value of its element
    type. May be used to load and store data in distinct memory slots.
  }];

  let mnemonic = "ptr";
  let parameters = (ins "::mlir::Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";

  let builders = [
    AttrBuilderWithInferredContext<(ins "::mlir::Type":$elementType), [{
      return $_get(elementType.getContext(), elementType);
    }]>
  ];
}

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

//===----------------------------------------------------------------------===//
// Type Constraints
//===----------------------------------------------------------------------===//

class LLHDPtrTypeOf<list<Type> allowedTypes>
  : ContainerType<AnyTypeOf<allowedTypes>, CPred<"llvm::isa<PtrType>($_self)">,
    "llvm::cast<PtrType>($_self).getElementType()", "LLHD pointer type">;

def LLHDAnySigOrPtrType : AnyTypeOf<[LLHDPtrType, InOutType]>;

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
