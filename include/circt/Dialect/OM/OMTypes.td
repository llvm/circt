//===- OMTypes.td - Object Model dialect types ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains the Object Model dialect type definitions.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_OM_OMTYPES_TD
#define CIRCT_DIALECT_OM_OMTYPES_TD

include "mlir/IR/AttrTypeBase.td"

def ClassType : TypeDef<OMDialect, "Class", []> {
  let summary = "A type that represents a reference to a Class.";

  let mnemonic = "class.type";

  let parameters = (ins
    "mlir::FlatSymbolRefAttr":$className
  );

  let assemblyFormat = [{
    `<` $className `>`
  }];
}

def ReferenceType : TypeDef<OMDialect, "Reference", []> {
  let summary = "A type that represents a reference to a hardware entity.";

  let mnemonic = "ref";
}

def ListType : TypeDef<OMDialect, "List", []> {
  let summary = "A type that represents a list.";

  let mnemonic = "list";
  let parameters = (ins "mlir::Type":$elementType);
  let assemblyFormat = [{
    `<` $elementType `>`
  }];

  let builders = [
    AttrBuilderWithInferredContext<(ins "::mlir::Type":$elementType), [{
      return $_get(elementType.getContext(), elementType);
    }]>
  ];
}

def SymbolRefType : TypeDef<OMDialect, "SymbolRef", []> {
  let summary = "A type that represents a reference to a flat symbol reference.";

  let mnemonic = "sym_ref";
}

def OMIntegerType : TypeDef<OMDialect, "OMInteger", []> {
  let summary = "A type that represents an arbitrary width integer.";

  let mnemonic = "integer";
}

def StringType : TypeDef<OMDialect, "String", []> {
  let summary = "A type that represents a string.";

  let mnemonic = "string";
}

def BasePathType : TypeDef<OMDialect, "BasePath", []> {
  let summary = "A fragment of a path to a hardware component";
  let mnemonic = "basepath";
}

def PathType : TypeDef<OMDialect, "Path", []> {
  let summary = "A path to a hardware component";
  let mnemonic = "path";
}

def FrozenBasePathType : TypeDef<OMDialect, "FrozenBasePath", []> {
  let summary = "A frozen fragment of a path to a hardware component";
  let mnemonic = "frozenbasepath";
}

def FrozenPathType : TypeDef<OMDialect, "FrozenPath", []> {
  let summary = "A frozen path to a hardware component";
  let mnemonic = "frozenpath";
}

def OMAnyType : TypeDef<OMDialect, "Any", []> {
  let summary = "A type that represents any valid OM type.";

  let mnemonic = "any";
}

#endif // CIRCT_DIALECT_OM_OMTYPES_TD
