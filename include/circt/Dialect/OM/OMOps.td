//===- OMOps.td - Object Model dialect ops --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains the Object Model dialect operation definitions.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_OM_OMOPS_TD
#define CIRCT_DIALECT_OM_OMOPS_TD

include "circt/Dialect/OM/OMDialect.td"
include "circt/Dialect/OM/OMEnums.td"
include "circt/Dialect/OM/OMOpInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"

class OMOp<string mnemonic, list<Trait> traits = []> :
    Op<OMDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Shared definitions
//===----------------------------------------------------------------------===//

class OMClassLike<string mnemonic, list<Trait> traits = []> :
  OMOp<mnemonic, traits # [
    SingleBlock, NoTerminator, Symbol, RegionKindInterface,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
    DeclareOpInterfaceMethods<ClassLike>]> {

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    StrArrayAttr:$formalParamNames
  );

  let regions = (region
    SizedRegion<1>:$body
  );

  let builders = [
    OpBuilder<(ins "::mlir::Twine":$name)>,
    OpBuilder<(ins "::mlir::Twine":$name,
                   "::mlir::ArrayRef<::mlir::StringRef>":$formalParamNames)>,
  ];

  let hasCustomAssemblyFormat = 1;

  let hasVerifier = 1;
}

class OMClassFieldLike<string mnemonic, list<Trait> traits = []> :
  OMOp<mnemonic, traits # [
    Symbol,
    DeclareOpInterfaceMethods<ClassFieldLike>]> {
}

//===----------------------------------------------------------------------===//
// Class definitions
//===----------------------------------------------------------------------===//

def ClassOp : OMClassLike<"class"> {
  let extraClassDeclaration = [{
    mlir::Block *getBodyBlock() { return &getBody().front(); }
    // This builds a ClassOp, and populates it with the CLassFieldOps.
    // Build the ClassOp with `name` and `formalParamNames`. Then add
    // ClassFieldOps for each name and type in `fieldNames` and `fieldTypes`.
    circt::om::ClassOp static buildSimpleClassOp(
    mlir::OpBuilder &odsBuilder, mlir::Location loc, mlir::Twine name,
    mlir::ArrayRef<mlir::StringRef> formalParamNames,
    mlir::ArrayRef<mlir::StringRef> fieldNames,
    mlir::ArrayRef<mlir::Type> fieldTypes);

    // Implement RegionKindInterface.
    static mlir::RegionKind getRegionKind(unsigned index) {
      return mlir::RegionKind::Graph;
    }
  }];
}

def ClassFieldOp : OMClassFieldLike<"class.field",
    [HasParent<"ClassOp">]> {
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    AnyType:$value
  );

  let assemblyFormat = [{
    $sym_name `,` $value  attr-dict `:` type($value)
  }];
}

//===----------------------------------------------------------------------===//
// External class definitions
//===----------------------------------------------------------------------===//

def ClassExternOp : OMClassLike<"class.extern"> {
  let extraClassDeclaration = [{
    mlir::Block *getBodyBlock() { return &getBody().front(); }
    
    // Implement RegionKindInterface.
    static mlir::RegionKind getRegionKind(unsigned index) {
      return mlir::RegionKind::Graph;
    }
  }];
}

def ClassExternFieldOp : OMClassFieldLike<"class.extern.field",
    [HasParent<"ClassExternOp">]> {
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttr:$type
  );

  let assemblyFormat = [{
    $sym_name attr-dict `:` $type
  }];
}

//===----------------------------------------------------------------------===//
// Object instantiations and fields
//===----------------------------------------------------------------------===//

def ObjectOp : OMOp<"object",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>, Pure]> {
  let arguments = (ins
    SymbolNameAttr:$className,
    Variadic<AnyType>:$actualParams
  );

  let results = (outs
    ClassType:$result
  );

  let builders = [
    OpBuilder<(ins "om::ClassOp":$classOp, "::mlir::ValueRange":$actualParams)>
  ];

  let assemblyFormat = [{
    $className `(` $actualParams `)`  attr-dict `:`
      functional-type($actualParams, $result)
  }];
}

def ObjectFieldOp : OMOp<"object.field",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>, Pure]> {
  let arguments = (ins
    ClassType:$object,
    FlatSymbolRefArrayAttr:$fieldPath
  );

  let results = (outs
    AnyType:$result
  );

  let assemblyFormat = [{
    $object `,` $fieldPath attr-dict `:` functional-type($object, $result)
  }];
}

//===----------------------------------------------------------------------===//
// Primitives and containers
//===----------------------------------------------------------------------===//

def ConstantOp : OMOp<"constant",
    [ConstantLike, Pure, AllTypesMatch<["value", "result"]>]> {
  let arguments = (ins
    TypedAttrInterface:$value
  );

  let results = (outs
    AnyType:$result
  );

  let builders = [
    OpBuilder<(ins "::mlir::TypedAttr":$constVal)>
  ];

  let assemblyFormat = [{
    $value attr-dict
  }];

  let hasFolder = true;
}

def ListCreateOp : OMOp<"list_create", [Pure, SameTypeOperands]> {
  let summary = "Create a list of values";
  let description = [{
    Creates a list from a sequence of inputs.

    ```
    %list = om.list_create %a, %b, %c : !om.ref
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs
    ListType:$result
  );

  let hasCustomAssemblyFormat = 1;
}

def TupleCreateOp : OMOp<"tuple_create", [Pure, InferTypeOpInterface]> {
  let summary = "Create a tuple of values";
  let description = [{
    Create a tuple from a sequence of inputs.

    ```
    %tuple = om.tuple_create %a, %b, %c : !om.ref, !om.string, !om.list<i32>
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs
    TupleOf<[AnyType]>:$result
  );

  let assemblyFormat = [{
    $inputs attr-dict `:` type($inputs)
  }];

  let extraClassDeclaration = [{
    // Implement InferTypeOpInterface.
    static ::mlir::LogicalResult inferReturnTypes(
      ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
      ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
      ::mlir::OpaqueProperties,
      ::mlir::RegionRange regions,
      ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes);
   }];

}

def TupleGetOp : OMOp<"tuple_get", [Pure, InferTypeOpInterface]> {
  let summary = "Extract a value from a tuple";
  let description = [{
    Extract a value from a tuple.

    ```
    %value = om.tuple_get %a[0] : tuple<!om.ref, !om.string, !om.list<i32>>
    ```
  }];

  let arguments = (ins
    TupleOf<[AnyType]>:$input,
    I32Attr:$index
  );

  let results = (outs
    AnyType:$result
  );

  let assemblyFormat = [{
    $input `[` $index `]` attr-dict `:` type($input)
  }];

  let extraClassDeclaration = [{
    // Implement InferTypeOpInterface.
    static ::mlir::LogicalResult inferReturnTypes(
      ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
      ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
      ::mlir::OpaqueProperties, ::mlir::RegionRange regions,
      ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes);
   }];
}

def MapKeyValuePair: Type<CPred<"::circt::om::isMapKeyValuePairType($_self)">,
                        "a pair whose first element is an attribute",
                        "::mlir::TupleType">;

def MapCreateOp : OMOp<"map_create", [Pure, SameTypeOperands]> {
  let summary = "Create a map";
  let description = [{
    Creates a map from a sequence of inputs.

    ```
    %map = om.map_create %e1, %e2 : !om.string, i8
    ```
    where `%e1` and `e2` have !om.tuple<!om.string, i8> and
    `%map` has `!om.map<!om.string, i8>` type.
  }];

  let arguments = (ins Variadic<MapKeyValuePair>:$inputs);
  let results = (outs
    MapType:$result
  );

  let hasCustomAssemblyFormat = true;
}

def PathOp : OMOp<"path", [Pure, 
      DeclareOpInterfaceMethods<SymbolUserOpInterface>
    ]> {
  let summary = "Produce a path value";
  let description = [{
    Produces a value which represents a hierarchical path to a hardware
    component.

    Example:
    ```mlir
    hw.module @Foo {
      %wire = hw.wire sym @w: !i1
    }
    hw.hierpath @Path [@Foo::@w]
    %0 = om.path member @Path
    ```
  }];
  let arguments = (ins TargetKind:$targetKind,
                       FlatSymbolRefAttr:$target);
  let results = (outs PathType:$result);
  let assemblyFormat = "$targetKind $target attr-dict";
}

def AnyCastOp : OMOp<"any_cast", [Pure]> {
  let summary = "Cast any value to any type.";

  let description = [{
    Casts any value to AnyType. This is useful for situations where a value of
    AnyType is needed, but a value of some concrete type is known.

    In the evaluator, this is a noop, and the value of concrete type is used.
  }];

  let arguments = (ins AnyType:$input);

  let results = (outs OMAnyType:$result);

  let assemblyFormat =
     "$input attr-dict `:` functional-type($input, $result)";
}

def PathAppendOp : OMOp<"path_append", [Pure]> {
  let summary = "Append two path value";
  let description = [{
    Appends the path to the root path. The root path must not have a component.
    The appended path must begin where the root path ends.

    Example:
    ```mlir
    %root = om.constant #om.path<"Foo/bar:Bar">
    %path = om.constant #om.path<"Bar/baz:Baz>output">
    %0 = om.path_append %root, %path
    ```
  }];
  let arguments = (ins PathType:$root, PathType:$path);
  let results = (outs PathType:$result);
  let assemblyFormat = "$root `,` $path attr-dict";
}
#endif // CIRCT_DIALECT_OM_OMOPS_TD
