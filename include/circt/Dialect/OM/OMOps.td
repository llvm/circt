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

include "circt/Dialect/OM/OMOpInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

class OMOp<string mnemonic, list<Trait> traits = []> :
    Op<OMDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Shared definitions
//===----------------------------------------------------------------------===//

class OMClassLike<string mnemonic, list<Trait> traits = []> :
  OMOp<mnemonic, traits # [
    SingleBlock, NoTerminator, Symbol, SymbolTable,
    HasParent<"mlir::ModuleOp">,
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
  }];
}

def ClassFieldOp : OMClassFieldLike<"class.field",
    [HasParent<"ClassOp">]> {
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    AnyType:$value
  );

  let assemblyFormat = [{
    $sym_name `,` $value `:` type($value) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// External class definitions
//===----------------------------------------------------------------------===//

def ClassExternOp : OMClassLike<"class.extern"> {
  let extraClassDeclaration = [{
    mlir::Block *getBodyBlock() { return &getBody().front(); }
  }];
}

def ClassExternFieldOp : OMClassFieldLike<"class.extern.field",
    [HasParent<"ClassExternOp">]> {
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttr:$type
  );

  let assemblyFormat = [{
    $sym_name `:` $type attr-dict
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
    $className `(` $actualParams `)` `:`
      functional-type($actualParams, $result) attr-dict
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
    $object `,` $fieldPath `:` functional-type($object, $result) attr-dict
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
    $inputs `:` type($inputs) attr-dict
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
    $input `[` $index `]` `:` type($input) attr-dict
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

#endif // CIRCT_DIALECT_OM_OMOPS_TD
