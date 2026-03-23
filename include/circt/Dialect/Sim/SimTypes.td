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

def DPIModuleType : SimTypeDef<"DPIModule"> {
  let summary = "DPI module type with first-class port directions";
  let description = [{
    Represents the full port signature of a DPI-C function declaration.
    Each port has a name, type, and direction (`input`, `output`, `inout`,
    `return`, or `ref`). Unlike `hw.modty`, this type preserves DPI-specific
    semantics such as explicit return values and by-reference ports.
  }];

  let mnemonic = "dpi_modty";
  let parameters = (ins
    ArrayRefParameter<"::circt::sim::DPIPort", "DPI port list">:$ports
  );
  let genStorageClass = 0;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Get the number of ports.
    size_t getNumPorts() const;

    /// Get the input ports (input, inout, ref — call operands).
    llvm::SmallVector<DPIPort> getInputPorts() const;

    /// Get the result ports (output, inout, return — call results).
    llvm::SmallVector<DPIPort> getResultPorts() const;

    /// Get the return port, if any. Returns nullptr if none.
    const DPIPort *getReturnPort() const;

    /// Derive the MLIR FunctionType for the call shape.
    mlir::FunctionType getFunctionType() const;

    /// Construct an hw::ModuleType for lowering to SV.
    ::circt::hw::ModuleType getHWModuleType() const;

    /// Verify structural constraints (return-is-last, at-most-one-return).
    /// On failure, emits diagnostics on the given op and returns failure.
    mlir::LogicalResult verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError) const;
  }];
}

#endif // CIRCT_DIALECT_SIM_SIMTYPES_TD
