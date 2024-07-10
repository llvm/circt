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
    In addition, the `dbg.variable` operation may contain extra information about 
    the variable, such as its source language type and eventual constructor 
    parameters. This is allows to reconstruct more precisely the source language.

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
  // let results = (outs VariableType:$result);
  let assemblyFormat = [{
    $name `,` $value (`scope` $scope^)? attr-dict (`enumDef` $enumDef^)? `:` type($value)
  }];
}

def SubFieldOp : DebugOp<"subfield"> {
  let summary = "A named value to be captured in debug info which is a subfield of an aggregate";
  let description = [{
    Marks a subfield of aggregate to be tracked in DI under the given name. 
    It is similar to `dbg.variable`, both store a value and contain source language name, type, 
    and constructor parameters, but `dbg.subfield` returns also a value. Unlike a `dbg.variable`, 
    it is contained in other debug operations like `dbg.struct` or `dbg.array` (here the usage of 
    the returned value). It is only used to represent a subfield of an aggregate and it cannot be
    used to represent a variable directly declared in a module.

    The addition of support for source language type and constructor parameters for top variables 
    and subfields (also nested) required to build this additional operation. The `dbg.variable` 
    explicitly represents the "top" variable instances in a module. For this reason, it wasn't used to mark
    and it cannot mark subfields of aggregates.

    The `dbg.subfield` doesn't have a `scope` operand, because it is a descendant of a `dbg.variable`.
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
    $name `,` $value attr-dict (`enumDef` $enumDef^)? `:` type($value)
  }];
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


def ModuleInfoOp : DebugOp<"moduleinfo"> {
  let summary = "Define extra debug information for a module";
  let description = [{
    Creates debug information for a module. If present, this operations provides
    extra information about the module type in the source language, such as its
    type name and constructor parameters. 
  }];

  let arguments = (ins
    StrAttr:$typeName,
    OptionalAttr<ArrayAttr>:$params
  );

  let assemblyFormat = [{ attr-dict }];
}

def EnumDefOp : DebugOp<"enumdef"> {
  let summary = "Define the value variants of an enumeration";
  let description = [{
    Creates a definition of an enumeration type.
    It is useful to reconstruct the named variants of an enum from a raw value.
    Variants can be internally represented as a map (Int -> String).
    It might be possible that the user select the order and single raw values 
    of the variants. Therefore, it is not possible to use an array to map the
    named variants with the integer value.

    This operation is declared once per enum present in a scope.

    The result of this operation is used as operand in `dbg.variable` and
    `dbg.subfield` operations if they originate from an enum type.
  }];

  let arguments = (ins
    StrAttr:$enumTypeName,
    I16Attr:$id,
    DictionaryAttr:$variantsMap,
    Optional<ScopeType>:$scope
  );

  let results = (outs EnumDefType:$result);

  let assemblyFormat = [{
    $enumTypeName `,` `id` $id `,` $variantsMap (`scope` $scope^)? attr-dict
  }];
}

#endif // CIRCT_DIALECT_DEBUG_DEBUGOPS_TD
