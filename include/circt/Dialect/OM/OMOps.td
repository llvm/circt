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

def BasePathCreateOp : OMOp<"basepath_create", [Pure, 
      DeclareOpInterfaceMethods<SymbolUserOpInterface>
    ]> {
  let summary = "Produce a base path value";
  let description = [{
    Produces a value which represents a fragment of a hierarchical path to a
    target. Given a base path, extend it with the name of a module instance, to
    produce a new base path. The instance is identified via an NLA. Once the
    final verilog name of the instance is known, this op can be converted into
    a FrozenBasePathOp.

    Example:
    ```mlir
    hw.module @Foo() -> () {
      hw.inst "bar" sym @bar @Bar() -> ()
    }
    hw.hierpath @Path [@Foo::@bar]
    om.class @OM(%basepath: !om.basepath) {
      %0 = om.basepath_create %base @Path
    }
    ```
  }];
  let arguments = (ins BasePathType:$basePath, FlatSymbolRefAttr:$target);
  let results = (outs BasePathType:$result);
  let assemblyFormat = "$basePath $target attr-dict";
}

def PathCreateOp : OMOp<"path_create", [Pure,
      DeclareOpInterfaceMethods<SymbolUserOpInterface>
    ]> {
  let summary = "Produce a path value";
  let description = [{
    Produces a value which represents a hierarchical path to a hardware
    target.
     from a base path to a target.

    Example:
    ```mlir
    hw.module @Foo() -> () {
      %wire = hw.wire sym @w: !i1
    }
    hw.hierpath @Path [@Foo::@w]
    om.class @OM(%basepath: !om.basepath)
      %0 = om.path_create reference %basepath @Path
    }
    ```
  }];
  let arguments = (ins
    TargetKind:$targetKind,
    BasePathType:$basePath,
    FlatSymbolRefAttr:$target
  );
  let results = (outs PathType:$result);
  let assemblyFormat = "$targetKind $basePath $target attr-dict";
}

def EmptyPathOp : OMOp<"path_empty", [Pure]> {
  let summary = "Produce a path value to nothing";
  let description = [{
    Produces a value which represents a hierarchical path to nothing.

    Example:
    ```mlir
    om.class @OM()
      %0 = om.path_empty
    }
    ```
  }];
  let results = (outs PathType:$result);
  let assemblyFormat = "attr-dict";
}

def FrozenBasePathCreateOp : OMOp<"frozenbasepath_create", [Pure]> {
  let summary = "Produce a frozen base path value";
  let description = [{
    Produces a value which represents a fragment of a hierarchical path to a
    target.

    Example:
    ```mlir
    om.class @OM(%basepath: !om.basepath)
      %0 = om.frozenbasepath_create %basepath "Foo/bar:Bar/baz"
    }
    ```
  }];
  let arguments = (ins FrozenBasePathType:$basePath, OMPathAttr:$path);
  let results = (outs FrozenBasePathType:$result);
  let assemblyFormat = "$basePath custom<BasePathString>($path) attr-dict";
}

def FrozenPathCreateOp : OMOp<"frozenpath_create", [Pure]> {
  let summary = "Produce a frozen path value";
  let description = [{
    Produces a value which represents a hierarchical path to a hardware
    component from a base path to a target.

    Example:
    ```mlir
    om.class @OM(%basepath: !om.basepath)
      %0 = om.frozenpath_create reference %base "Foo/bar:Bar>w.a"
    }
    ```
  }];
  let arguments = (ins
    TargetKind:$targetKind,
    FrozenBasePathType:$basePath,
    OMPathAttr:$path,
    StrAttr:$module,
    StrAttr:$ref,
    StrAttr:$field
  );
  let results = (outs FrozenPathType:$result);
  let assemblyFormat = [{
    $targetKind $basePath custom<PathString>($path, $module, $ref, $field)
      attr-dict
  }];
}

def FrozenEmptyPathOp : OMOp<"frozenpath_empty", [Pure]> {
  let summary = "Produce a frozen path value to nothing";
  let description = [{
    Produces a value which represents a hierarchical path to nothing.

    Example:
    ```mlir
    om.class @OM()
      %0 = om.frozenpath_empty
    }
    ```
  }];
  let results = (outs FrozenPathType:$result);
  let assemblyFormat = "attr-dict";
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
#endif // CIRCT_DIALECT_OM_OMOPS_TD
