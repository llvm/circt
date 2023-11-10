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

include "mlir/Interfaces/ControlFlowInterfaces.td"
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
    Symbol, RegionKindInterface, IsolatedFromAbove,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
    DeclareOpInterfaceMethods<ClassLike>]> {

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    StrArrayAttr:$formalParamNames,
    ArrayAttr:$fieldNames,
    DictionaryAttr:$fieldTypes
  );

  let regions = (region
    SizedRegion<1>:$body
  );

  let builders = [
    OpBuilder<(ins "::mlir::Twine":$name,
                   "::mlir::ArrayRef<::mlir::StringRef>":$formalParamNames,
                   "::mlir::ArrayRef<::mlir::Attribute>":$fieldNames,
                   "::mlir::ArrayRef<::mlir::NamedAttribute>":$fieldTypes), [{
      build(odsBuilder, odsState, odsBuilder.getStringAttr(name),
            odsBuilder.getStrArrayAttr(formalParamNames),
            odsBuilder.getArrayAttr(fieldNames),
            odsBuilder.getDictionaryAttr(fieldTypes));
    }]>,
    OpBuilder<(ins "::mlir::Twine":$name,
                   "::mlir::ArrayRef<::mlir::StringRef>":$formalParamNames), [{
      build(odsBuilder, odsState, odsBuilder.getStringAttr(name),
            odsBuilder.getStrArrayAttr(formalParamNames));
    }]>,
    OpBuilder<(ins "::mlir::Twine":$name), [{
      build(odsBuilder, odsState, odsBuilder.getStringAttr(name),
            odsBuilder.getStrArrayAttr({}));
    }]>,
    OpBuilder<(ins "::mlir::Twine":$name,
                   "::mlir::ArrayAttr":$fieldNames,
                   "::mlir::DictionaryAttr":$fieldTypes), [{
      build(odsBuilder, odsState, odsBuilder.getStringAttr(name),
            odsBuilder.getStrArrayAttr({}), fieldNames, fieldTypes);
    }]>,
    OpBuilder<(ins "::mlir::Twine":$name,
                   "::mlir::ArrayRef<::mlir::StringRef>":$formalParamNames,
                   "::mlir::ArrayAttr":$fieldNames,
                   "::mlir::DictionaryAttr":$fieldTypes), [{
      build(odsBuilder, odsState, odsBuilder.getStringAttr(name),
            odsBuilder.getStrArrayAttr(formalParamNames), fieldNames,
            fieldTypes);
    }]>,
    OpBuilder<(ins "::mlir::StringAttr":$name,
                   "::mlir::ArrayAttr":$formalParamNames), [{
      build(odsBuilder, odsState, name, formalParamNames,
            odsBuilder.getArrayAttr({}), odsBuilder.getDictionaryAttr({}));

    }]>
  ];

  let hasCustomAssemblyFormat = 1;

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Class definitions
//===----------------------------------------------------------------------===//

def ClassOp : OMClassLike<"class", [
      SingleBlockImplicitTerminator<"ClassFieldsOp">]> {
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

    circt::om::ClassFieldsOp getFieldsOp() {
      return mlir::cast<ClassFieldsOp>(this->getBodyBlock()->getTerminator());
    }

    // The addNewFieldsOp API encapsulates the logic used to represent field
    // locations under the hood.  Users should invoke this method rather
    // than construct the operation directly, otherwise logic to retrieve
    // the field location will break.
    // This is required because MLIR's FusedLoc uses a "set" semantics where a
    // single location is used to represent multiple fields with the same
    // location.  The OM implementation uses an attribute to store the original
    // array of locations, so that the specific location of a field may be
    // easily retrieved by index using the `getFieldLocByIndex` API.
    void addNewFieldsOp(mlir::OpBuilder &builder, mlir::ArrayRef<mlir::Location>
                   locs, mlir::ArrayRef<mlir::Value> values);

    // Add additional fields to the class. This updates the ClassOp output type
    //  with the additional field types and field names and adds the field
    // values to the fieldsOp.
    void updateFields(mlir::ArrayRef<mlir::Location> newLocations,
                   mlir::ArrayRef<mlir::Value> newValues,
                   mlir::ArrayRef<mlir::Attribute> newNames);

    // Return the location for a field referenced by index in the fieldNames
    // array attribute.  If the field has a location added by addNewFieldsOp API,
    // its location will be retrieved from the array of per field locations.
    // Otherwise, it will inherit the location of the class op Using this with
    // a ClassFieldsOp that has been constructed with a FusedLoc but not
    // following the internal storage format of `addNewFieldsOp` will result in an
    // assertion error
    mlir::Location getFieldLocByIndex(size_t i);
  }];

  let hasRegionVerifier = 1;
}

def ClassFieldsOp : OMOp<"class.fields", [Terminator, ReturnLike, Pure,
      HasParent<"ClassOp">]> {
  let arguments = (ins Variadic<AnyType>:$fields, 
                       OptionalAttr<LocationArrayAttr>:$field_locs);
  let assemblyFormat = [{
    attr-dict ($fields^ `:` qualified(type($fields)))?
    custom<FieldLocs>($field_locs)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// External class definitions
//===----------------------------------------------------------------------===//

def ClassExternOp : OMClassLike<"class.extern", [NoTerminator]> {
  let extraClassDeclaration = [{
    mlir::Block *getBodyBlock() { return &getBody().front(); }

    // Implement RegionKindInterface.
    static mlir::RegionKind getRegionKind(unsigned index) {
      return mlir::RegionKind::Graph;
    }
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

def ObjectFieldOp : OMOp<"object.field", [Pure]> {
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

def ListConcatOp : OMOp<"list_concat", [Pure, SameOperandsAndResultType]> {
  let summary = "Concatenate multiple lists to produce a new list";
  let description = [{
    Produces a value of list type by concatenating the provided lists.

    Example:
    ```
    %3 = om.list_concat %0, %1, %2 : !om.list<string>
    ```
  }];

  let arguments = (ins Variadic<ListType>:$subLists);
  let results = (outs ListType:$result);

  let assemblyFormat = "$subLists attr-dict `:` type($result)";
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

class IntegerBinaryArithmeticOp<string mnemonic, list<Trait> traits = []> :
    OMOp<mnemonic, [
      Pure,
      DeclareOpInterfaceMethods<IntegerBinaryArithmeticInterface>
    ] # traits> {
  let arguments = (ins OMIntegerType:$lhs, OMIntegerType:$rhs);

  let results = (outs OMIntegerType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def IntegerAddOp : IntegerBinaryArithmeticOp<"integer.add", [Commutative]> {
  let summary = "Add two OMIntegerType values";
  let description = [{
    Perform arbitrary precision signed integer addition of two OMIntegerType
    values.

    Example:
    ```mlir
    %2 = om.integer.add %0, %1 : !om.integer
    ```
  }];
}

def IntegerMulOp : IntegerBinaryArithmeticOp<"integer.mul", [Commutative]> {
  let summary = "Multiply two OMIntegerType values";
  let description = [{
    Perform arbitrary prevision signed integer multiplication of two
    OMIntegerType values.

    Example:
    ```mlir
    %2 = om.integer.mul %0, %1 : !om.integer
    ```
  }];
}

def IntegerShrOp : IntegerBinaryArithmeticOp<"integer.shr"> {
  let summary = "Shift an OMIntegerType value right by an OMIntegerType value";
  let description = [{
    Perform arbitrary precision signed integer arithmetic shift right of the lhs
    OMIntegerType value by the rhs OMIntegerType value. The rhs value must be
    non-negative.

    Example:
    ```mlir
    %2 = om.integer.shr %0, %1 : !om.integer
    ```
  }];
}

def IntegerShlOp : IntegerBinaryArithmeticOp<"integer.shl"> {
  let summary = "Shift an OMIntegerType value left by an OMIntegerType value";
  let description = [{
    Perform arbitrary precision signed integer arithmetic shift left of the lhs
    OMIntegerType value by the rhs OMIntegerType value. The rhs value must be
    non-negative.

    Example:
    ```mlir
    %2 = om.integer.shl %0, %1 : !om.integer
    ```
  }];
}

#endif // CIRCT_DIALECT_OM_OMOPS_TD
