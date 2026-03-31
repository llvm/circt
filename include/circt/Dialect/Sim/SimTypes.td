//===- SimTypes.td - Sim dialect types ---------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SIM_SIMTYPES_TD
#define CIRCT_DIALECT_SIM_SIMTYPES_TD

include "circt/Dialect/Sim/SimDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

class SimTypeDef<string name> : TypeDef<SimDialect, name> { }

def FormatStringType : SimTypeDef<"FormatString"> {
  let mnemonic = "fstring";

  let summary = "Format string type";
  let description = [{
    A format string type represents either a single formatting fragment or the
    concatenation of an arbitrary but finite number of fragments.
    A formatting fragment is either a static string literal or the association
    of a dynamic hardware value with a format specifier.
  }];
}

def DynamicStringType : SimTypeDef<"DynamicString"> {
  let mnemonic = "dstring";

  let summary = "String type with variable length"; 
  let description = [{
    The dynamic string type represents a string with a variable length that is 
    only known at runtime. The contents of the string are held as an array of
    `i8` bytes. The interpretation of these bytes as a concrete character 
    encoding is undefined, but treating the bytes as a UTF-8 string is 
    recommended. A string's length must fit into an `i64`.
    
    Even though practical lowerings of this type likely have to hold the string
    contents in dynamically-allocated heap memory, *this type has value
    semantics*. If the same string SSA value is passed to multiple users, any
    mutation performed by one user is not visible to other users.
  }];
}

def QueueType : SimTypeDef<"Queue"> {
  let mnemonic = "queue";
  let summary = "a queue type"; 
  let description = [{
    An ordered container of elements with an optional upper bound 
    on the number of elements that it can hold. If this bound is 0, then the
    queue has unrestricted capacity.

    Even though practical lowerings of this type likely have to hold the queue
    contents in dynamically-allocated heap memory, *this type has value
    semantics*.
  }];

  let parameters = (ins
    "::mlir::Type":$elementType,
    "unsigned":$bound
  );
  let assemblyFormat = [{
    `<` $elementType `,` $bound `>`
  }];
  let builders = [
    AttrBuilderWithInferredContext<
      (ins "Type":$elementType, "unsigned":$bound), [{
        return $_get(elementType.getContext(), elementType, bound);
      }]>
  ];
}

def DPIFunctionType : SimTypeDef<"DPIFunction"> {
  let summary = "DPI function type with first-class argument directions";
  let description = [{
    Represents the full argument signature of a DPI-C function declaration.
    Each argument has a name, type, and direction (`input`, `output`, `inout`,
    `return`, or `ref`). Unlike `hw.modty`, this type preserves DPI-specific
    semantics such as explicit return values and by-reference arguments.
  }];

  let mnemonic = "dpi_functy";
  let parameters = (ins
    ArrayRefParameter<"::circt::sim::DPIArgument", "DPI argument list">:$arguments
  );
  let genStorageClass = 0;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Get the number of arguments.
    size_t getNumArguments() const;

    /// Get the input arguments (input, inout, ref — call operands).
    llvm::SmallVector<DPIArgument> getInputArguments() const;

    /// Get the result arguments (output, inout, return — call results).
    llvm::SmallVector<DPIArgument> getResultArguments() const;

    /// Get the return argument, if any. Returns nullptr if none.
    const DPIArgument *getReturnArgument() const;

    /// Derive the MLIR FunctionType for the call shape.
    mlir::FunctionType getFunctionType() const;

    /// Verify structural constraints (return-is-last, at-most-one-return).
    /// On failure, emits diagnostics on the given op and returns failure.
    mlir::LogicalResult verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError) const;
  }];
}

#endif // CIRCT_DIALECT_SIM_SIMTYPES_TD
