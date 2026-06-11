//===- DebugOps.td - Debug dialect operations --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_DEBUG_DEBUGOPS_TD
#define CIRCT_DIALECT_DEBUG_DEBUGOPS_TD

include "circt/Dialect/Debug/DebugDialect.td"
include "circt/Dialect/Debug/DebugTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

class DebugOp<string mnemonic, list<Trait> traits = []> :
  Op<DebugDialect, mnemonic, traits>;


def ScopeOp : DebugOp<"scope"> {
  let summary = "Define a scope for debug values";
  let description = [{
    Creates an additional level of hierarchy in the DI, a "scope", which can be
    used to group variables and other scopes.

    Operations such as `hw.module` introduce an implicit scope. All debug
    operations within a module are added to that implicit scope, unless they
    have an explicit `scope` operand. Providing an explicit scope can be used to
    represent inlined modules.

    Scopes in DI do not necessarily have to correspond to levels of a module
    hierarchy. They can also be used to model things like control flow scopes,
    call stacks, and other source-language concepts.

    The `scope` operand of any debug dialect operation must be defined locally
    by a `dbg.scope` operation. It cannot be a block argument. (This is intended
    as a temporary restriction, to be lifted in the future.)
  }];
  let arguments = (ins
    StrAttr:$instanceName,
    StrAttr:$moduleName,
    Optional<ScopeType>:$scope
  );
  let results = (outs ScopeType:$result);
  let assemblyFormat = [{
    $instanceName `,` $moduleName (`scope` $scope^)? attr-dict
  }];
}


def VariableOp : DebugOp<"variable", [AttrSizedOperandSegments]> {
  let summary = "A named value to be captured in debug info";
  let description = [{
    Marks a value to be tracked in DI under the given name. The `dbg.variable`
    operation is useful to represent named values in a source language. For
    example, ports, constants, parameters, variables, nodes, or name aliases can
    all be represented as a variable. In combination with `dbg.array` and
    `dbg.struct`, complex aggregate source language values can be described and
    reconstituted from individual IR values. The `dbg.variable` operation acts
    as a tracker that follows the evolution of its assigned value throughout the
    compiler's pass pipelines. The debug info analysis uses this op to populate
    a module's scope with named source language values, and to establish how
    these source language values can be reconstituted from the actual IR values
    present at the end of compilation.

    The optional `typeName` attribute and `params` array attribute carry
    source-language type information (the type name and constructor parameters),
    letting consumers reconstruct the source language type more precisely. The
    optional `enumDef` operand links the variable to a `dbg.enumdef` that
    describes the named variants of an enumeration type.

    See the rationale for examples and details. See the `dbg.scope` operation
    for additional details on how to use the `scope` operand.
  }];
  let arguments = (ins
    StrAttr:$name,
    AnyType:$value,
    OptionalAttr<StrAttr>:$typeName,
    OptionalAttr<ArrayAttr>:$params,
    Optional<EnumDefType>:$enumDef,
    Optional<ScopeType>:$scope
  );
  let assemblyFormat = [{
    $name `,` $value (`typeName` $typeName^)? (`params` $params^)?
        (`enumDef` $enumDef^)? (`scope` $scope^)? attr-dict `:` type($value)
  }];

  let builders = [
    OpBuilder<(ins "::mlir::StringAttr":$name, "::mlir::Value":$value,
                   "::mlir::Value":$scope), [{
      build($_builder, $_state, name, value, /*typeName=*/nullptr,
            /*params=*/nullptr, /*enumDef=*/nullptr, scope);
    }]>
  ];
}


def SubFieldOp : DebugOp<"subfield"> {
  let summary = "A named aggregate-leaf value to be captured in debug info";
  let description = [{
    Tags a single leaf of an aggregate for debug info, carrying the same
    optional source-language type / enum metadata as `dbg.variable`. Unlike
    `dbg.variable`, a `dbg.subfield` is intended to appear only as an operand
    of a `dbg.struct` or `dbg.array`. Freshly-created ops with no users are
    tolerated by the verifier to accommodate construction patterns; any
    surviving user must be `dbg.struct` or `dbg.array`.
  }];
  let arguments = (ins
    StrAttr:$name,
    AnyType:$value,
    OptionalAttr<StrAttr>:$typeName,
    OptionalAttr<ArrayAttr>:$params,
    Optional<EnumDefType>:$enumDef
  );
  let results = (outs SubFieldType:$result);
  let assemblyFormat = [{
    $name `,` $value (`typeName` $typeName^)? (`params` $params^)?
        (`enumDef` $enumDef^)? attr-dict `:` type($value)
  }];
  let hasVerifier = 1;
}

def ModuleInfoOp : DebugOp<"moduleinfo"> {
  let summary = "Define extra debug information for a module";
  let description = [{
    Creates debug information for a module. If present, this operation provides
    extra information about the module type in the source language, such as its
    type name and constructor parameters.

    At most one `dbg.moduleinfo` may appear in a given region.
  }];

  let arguments = (ins
    StrAttr:$typeName,
    OptionalAttr<ArrayAttr>:$params
  );

  let assemblyFormat = [{
    `typeName` $typeName (`params` $params^)? attr-dict
  }];

  let hasVerifier = 1;
}

def EnumDefOp : DebugOp<"enumdef"> {
  let summary = "Define the value variants of an enumeration";
  let description = [{
    Defines an enumeration type so consumers can reconstruct the named
    variants from a raw integer value. Logically the mapping is
    `int -> name`, but in IR it is stored as a `DictionaryAttr` keyed by
    variant name with `IntegerAttr` values (the integer is not a valid
    dictionary key); consumers reading variants programmatically must
    invert the map. `DictionaryAttr` sorts keys lexicographically, so
    declaration order is not preserved and consumers must not rely on it.

    Declared once per enum per scope. The result is consumed by
    `dbg.variable` and `dbg.subfield` ops whose value has this enum type.

    Intentionally not marked `Pure` (or similar): even an `enumdef` with no
    `dbg.variable` / `dbg.subfield` users must survive DCE because
    `DebugInfoBuilder` assigns sequential IDs to every enumdef in walk
    order, and downstream consumers may reference those IDs via metadata
    that the analysis cannot see.
  }];

  let arguments = (ins
    StrAttr:$enumTypeName,
    StrAttr:$fqn,
    DictionaryAttr:$variantsMap,
    Optional<ScopeType>:$scope
  );

  let results = (outs EnumDefType:$result);

  let assemblyFormat = [{
    $enumTypeName `,` `fqn` $fqn `,` $variantsMap (`scope` $scope^)? attr-dict
  }];

  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

def StructOp : DebugOp<"struct", [
  Pure,
  PredOpTrait<"number of fields and names match",
    CPred<"$fields.size() == $names.size()">>
]> {
  let summary = "Aggregate values into a struct";
  let description = [{
    Creates a struct aggregate from a list of names and values. The `dbg.struct`
    operation allows for struct-like source language values to be captured in
    the debug info. This includes structs, unions, bidirectional bundles,
    interfaces, classes, and other similar structures.

    See the rationale for examples and details.
  }];
  let arguments = (ins Variadic<AnyType>:$fields, StrArrayAttr:$names);
  let results = (outs StructType:$result);
  let hasCustomAssemblyFormat = 1;
}


def ArrayOp : DebugOp<"array", [Pure, SameTypeOperands]> {
  let summary = "Aggregate values into an array";
  let description = [{
    Creates an array aggregate from a list of values. The first operand is
    placed at array index 0. The last operand is placed at the highest array
    index. The `dbg.array` operation allows for array-like source language
    values to be captured in the debug info. This includes arrays, or in the
    case of SystemVerilog, packed and unpacked arrays, lists, sequences, queues,
    FIFOs, channels, and vectors.

    See the rationale for examples and details.
  }];
  let arguments = (ins Variadic<AnyType>:$elements);
  let results = (outs ArrayType:$result);
  let hasCustomAssemblyFormat = 1;
}

#endif // CIRCT_DIALECT_DEBUG_DEBUGOPS_TD
