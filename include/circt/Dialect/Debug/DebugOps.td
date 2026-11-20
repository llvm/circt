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
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

class DebugOp<string mnemonic, list<Trait> traits = []> :
  Op<DebugDialect, mnemonic, traits>;

def PortDirection : I32EnumAttr<"PortDirection", "the direction of a port", [
      I32EnumAttrCase<"In", 0, "in">,
      I32EnumAttrCase<"Out", 1, "out">,
      I32EnumAttrCase<"InOut", 2, "inout">
    ]> {
  let cppNamespace = "circt::debug";
}

def VariableKind : I32EnumAttr<"VariableKind",
    "what the source declares a variable as", [
      I32EnumAttrCase<"Reg", 0, "reg">,
      I32EnumAttrCase<"Mem", 1, "mem">,
      I32EnumAttrCase<"Node", 2, "node">,
      I32EnumAttrCase<"Wire", 3, "wire">
    ]> {
  let cppNamespace = "circt::debug";
}

def ValueKind : I32EnumAttr<"ValueKind",
    "the kind of a value that lowering erases", [
      I32EnumAttrCase<"Clock", 0, "clock">,
      I32EnumAttrCase<"Reset", 1, "reset">,
      I32EnumAttrCase<"AsyncReset", 2, "asyncreset">,
      I32EnumAttrCase<"SInt", 3, "sint">
    ]> {
  let cppNamespace = "circt::debug";
}


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

    For an inlined instance, `typeName` and `ext` keep its class,
    as on `#dbg.instance`. After dedup the scope can carry the `dbg.instance`
    list of the instance, and each merged module reads its own entry there.
  }];
  let arguments = (ins
    StrAttr:$instanceName,
    StrAttr:$moduleName,
    Optional<ScopeType>:$scope,
    OptionalAttr<StrAttr>:$typeName,
    OptionalAttr<DictionaryAttr>:$ext
  );
  let results = (outs ScopeType:$result);
  let assemblyFormat = [{
    $instanceName `,` $moduleName (`scope` $scope^)? (`typeName` $typeName^)?
        (`ext` $ext^)? attr-dict
  }];
  let builders = [
    OpBuilder<(ins "::mlir::StringAttr":$instanceName,
                   "::mlir::StringAttr":$moduleName, "::mlir::Value":$scope), [{
      build($_builder, $_state, instanceName, moduleName, scope, StringAttr{},
            DictionaryAttr{});
    }]>,
    OpBuilder<(ins "::llvm::StringRef":$instanceName,
                   "::llvm::StringRef":$moduleName, "::mlir::Value":$scope), [{
      build($_builder, $_state, $_builder.getStringAttr(instanceName),
            $_builder.getStringAttr(moduleName), scope);
    }]>
  ];
  let hasVerifier = 1;
}


def VariableOp : DebugOp<"variable"> {
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

    An enum-typed variable embeds a `dbg.enum` value, which carries the named
    variants. Source-language type information is not carried on the variable
    itself; wrap the value in a `dbg.value` to attach it.

    The optional `kind` is what the source declares the variable as: `wire`,
    `reg`, `mem`, or `node` for a named combinational expression. An
    optimized-away register stays `reg`. Without `kind` the source did not say.

    The optional `dir` marks a port: `in`, `out` or `inout` (also analog).
    For an aggregate it is the direction of its unflipped fields. A `reg` may
    be a port, a `mem` or `node` may not.

    See the rationale for examples and details. See the `dbg.scope` operation
    for additional details on how to use the `scope` operand.
  }];
  let arguments = (ins
    StrAttr:$name,
    AnyType:$value,
    Optional<ScopeType>:$scope,
    OptionalAttr<VariableKind>:$kind,
    OptionalAttr<PortDirection>:$dir
  );
  let assemblyFormat = [{
    $name `,` $value (`scope` $scope^)? ($kind^)? (`dir` $dir^)? attr-dict
        `:` type($value)
  }];
  let hasVerifier = 1;
  let builders = [
    // A variable with no kind and no direction.
    OpBuilder<(ins "::mlir::StringAttr":$name, "::mlir::Value":$value,
                   "::mlir::Value":$scope), [{
      build($_builder, $_state, name, value, scope, VariableKindAttr{},
            PortDirectionAttr{});
    }]>,
    OpBuilder<(ins "::llvm::StringRef":$name, "::mlir::Value":$value,
                   "::mlir::Value":$scope), [{
      build($_builder, $_state, $_builder.getStringAttr(name), value, scope);
    }]>
  ];
}


def ValueOp : DebugOp<"value"> {
  let summary = "Attach source-language metadata to a value";
  let description = [{
    Annotates an SSA value with additional source-language information: the
    source type name, language-specific data, and a dedicated location.
    The result is a metadata wrapper accepted wherever other debug operations
    accept plain values (`dbg.variable`, `dbg.struct`, `dbg.array`); consumers
    unwrap it to reach the underlying value. Keeping the metadata on a
    separate wrapper keeps `dbg.variable` and the aggregate ops orthogonal to
    source-language type information.

    An enum-typed value embeds a `dbg.enum`, so no enum linkage is carried on
    the wrapper itself. The optional `kind` keeps what lowering erases:
    `clock`, `reset`, `asyncreset` or `sint`. A clock or reset of builtin
    integer type is `i1`.

    The optional `ext` maps a lowercase language name to a JSON string.
    The verifier does not parse it.
    Freshly-created ops with no users are tolerated by the verifier to
    accommodate construction patterns; any surviving user must be a
    debug-dialect op.
  }];
  let arguments = (ins
    AnyType:$value,
    OptionalAttr<ValueKind>:$kind,
    OptionalAttr<StrAttr>:$typeName,
    OptionalAttr<DictionaryAttr>:$ext
  );
  let results = (outs ValueType:$result);
  let assemblyFormat = [{
    $value (`typeName` $typeName^)? (`ext` $ext^)?
        ($kind^)? attr-dict `:` type($value)
  }];
  let hasVerifier = 1;
}

def EnumOp : DebugOp<"enum", [Pure]> {
  let summary = "Interpret an integer value as an enumeration";
  let description = [{
    Wraps an integer value as a tag-only (C-style) enumeration, producing an
    enum-typed debug value. Behaves like a cast: the `value` operand is the raw
    integer tag, and `variantsMap` provides the `name -> tag` mapping consumers
    use to reconstruct the named variant. The result is an ordinary value that
    can be embedded in `dbg.variable`, `dbg.struct`, and `dbg.array` just like
    any other debug value, so enum tracking needs no separate definition op,
    scope, or subfield linkage.

    `enumTypeName` is the source-language type name. The optional `ext` is as
    on `dbg.value`.

    This models tag-only enums: `variantsMap` is keyed by variant name with
    `IntegerAttr` tags and has no slot for per-variant payload types, so
    data-carrying variants are out of scope.
  }];
  let arguments = (ins
    AnyType:$value,
    StrAttr:$enumTypeName,
    DictionaryAttr:$variantsMap,
    OptionalAttr<DictionaryAttr>:$ext
  );
  let results = (outs EnumType:$result);
  let assemblyFormat = [{
    $value `,` $enumTypeName `,` $variantsMap (`ext` $ext^)?
        attr-dict
        `:` type($value)
  }];
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

    The optional `flips` marks, per field, a direction opposite to the
    struct's, like a FIRRTL `flip`. Nested flips compose with the variable's
    `dir`. A `flips` with no flipped field is invalid; leave it out.

    See the rationale for examples and details.
  }];
  let arguments = (ins
    Variadic<AnyType>:$fields,
    StrArrayAttr:$names,
    OptionalAttr<DenseBoolArrayAttr>:$flips
  );
  let results = (outs StructType:$result);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let builders = [
    // A struct with no flipped fields.
    OpBuilder<(ins "::mlir::ValueRange":$fields, "::mlir::ArrayAttr":$names), [{
      build($_builder, $_state, fields, names, DenseBoolArrayAttr{});
    }]>,
    // A struct with `flips`, left out if no field is flipped.
    OpBuilder<(ins "::mlir::ValueRange":$fields, "::mlir::ArrayAttr":$names,
                   "::llvm::ArrayRef<bool>":$flips), [{
      build($_builder, $_state, fields, names,
            ::llvm::is_contained(flips, true)
                ? $_builder.getDenseBoolArrayAttr(flips)
                : DenseBoolArrayAttr{});
    }]>
  ];
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
