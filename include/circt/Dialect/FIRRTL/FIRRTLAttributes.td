//===- FIRRTLAttributes.td - FIRRTL Types Definition -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions of FIRRTL attributes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIRRTLATTRIBUTES_TD
#define CIRCT_DIALECT_FIRRTL_FIRRTLATTRIBUTES_TD

include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// FIRRTL Annotations Definition
//===----------------------------------------------------------------------===//

def AnnotationArrayAttr: ArrayAttrBase<
    And<[
      // Guarantee this is an ArrayAttr first
      CPred<"$_self.isa<::mlir::ArrayAttr>()">,
      // Guarantee all elements are DictionaryAttr or SubAnnotationAttr
      CPred<"::llvm::all_of($_self.cast<::mlir::ArrayAttr>(), "
            "[&](::mlir::Attribute attr) { return attr.isa<"
            "::mlir::DictionaryAttr,"
            "::circt::firrtl::SubAnnotationAttr>();})">]>,
    "Annotation array attribute"> {
  let constBuilderCall = "$_builder.getArrayAttr($0)";
}

def SubAnnotationAttr : AttrDef<FIRRTLDialect, "SubAnnotation"> {
  let summary = "An Annotation that targets part of what it's attached to";
  let description = [{
    An Annotation that is only applicable to part of what it is attached to.
    This uses a field ID to indicate to which field it is applicable.
  }];
  let mnemonic = "subAnno";
  let parameters = (ins "int64_t":$fieldID, "DictionaryAttr":$annotations);

  let assemblyFormat = "`<` `fieldID` `=` $fieldID `,` $annotations `>`";
}

def PortAnnotationsAttr : ArrayAttrBase<
    And<[
      // Guarantee this is an ArrayAttr first
      CPred<"$_self.isa<::mlir::ArrayAttr>()">,
      // Guarantee all elements are DictionaryAttr or SubAnnotationAttr
      CPred<"::llvm::all_of($_self.cast<::mlir::ArrayAttr>(), "
            "[&](::mlir::Attribute attr) { return attr.isa<"
            "::mlir::ArrayAttr,"
            "::mlir::DictionaryAttr>();})">]>,
    "Port annotations attribute"> {
  let constBuilderCall = "$_builder.getArrayAttr($0)";
}

def InvalidValueAttr : AttrDef<FIRRTLDialect, "InvalidValue"> {
  let summary = "A constant value of firrtl.invalid type";
  let description = [{
    Represents an firrtl.invalidvalue value, whose type is specified by the
    type of the attribute.
  }];
  let mnemonic = "invalidvalue";
  let parameters = (ins  AttributeSelfTypeParameter<"">:$type);

  let assemblyFormat = "";

  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type),
                                   "return $_get(type.getContext(), type);">
  ];
}

//===----------------------------------------------------------------------===//
// Grand Central-related Annotations
//===----------------------------------------------------------------------===//

class AugmentedType<string name> : AttrDef<FIRRTLDialect, name> {
  let description = "Used in the GrandCentralPass.";
  let parameters = (
    ins "DictionaryAttr":$underlying
  );

  code defaultClassDeclaration = [{
    StringAttr getClass() { return getUnderlying().getAs<StringAttr>("class"); }
  }];

  code hasName = [{
    StringAttr getName() { return getUnderlying().getAs<StringAttr>("name"); }
  }];

  code hasDefName = [{
    StringAttr getDefName() { return getUnderlying().getAs<StringAttr>("defName"); }
  }];

  code hasID = [{
    IntegerAttr getID() { return getUnderlying().getAs<IntegerAttr>("id"); }
  }];

  code hasElements = [{
    ArrayAttr getElements() { return getUnderlying().getAs<ArrayAttr>("elements"); }
  }];

  code hasPrefix = [{
    StringAttr getPrefix() { return getUnderlying().getAs<StringAttr>("prefix"); }
  }];

}

def AugmentedBundleType : AugmentedType<"AugmentedBundleType"> {
  let summary = "GrandCentral AugmentedBundleType";
  let extraClassDeclaration =
    defaultClassDeclaration #
    hasID #
    hasElements #
    hasDefName #
    hasPrefix # [{
      bool isRoot() { return getID() != nullptr; }
    }];
}

def AugmentedVectorType : AugmentedType<"AugmentedVectorType"> {
  let summary = "GrandCentral AugmentedVectorType";
  let extraClassDeclaration = defaultClassDeclaration # hasElements;
}

def AugmentedGroundType : AugmentedType<"AugmentedGroundType"> {
  let summary = "GrandCentral AugmentedGroundType";
  let extraClassDeclaration = hasID # hasName;
}

def AugmentedStringType : AugmentedType<"AugmentedStringType"> {
  let summary = "GrandCentral AugmentedStringType";
  let extraClassDeclaration = hasName;
}
def AugmentedBooleanType : AugmentedType<"AugmentedBooleanType"> {
  let summary = "GrandCentral AugmentedBooleanType";
  let extraClassDeclaration = hasName;
}
def AugmentedIntegerType : AugmentedType<"AugmentedIntegerType"> {
  let summary = "GrandCentral AugmentedIntegerType";
  let extraClassDeclaration = hasName;
}
def AugmentedDoubleType : AugmentedType<"AugmentedDoubleType"> {
  let summary = "GrandCentral AugmentedDoubleType";
  let extraClassDeclaration = hasName;
}
def AugmentedLiteralType : AugmentedType<"AugmentedLiteralType"> {
  let summary = "GrandCentral AugmentedLiteralType";
  let extraClassDeclaration = hasName;
}
def AugmentedDeletedType : AugmentedType<"AugmentedDeletedType"> {
  let summary = "GrandCentral AugmentedDeletedType";
  let extraClassDeclaration = hasName;
}


def ParamDeclAttr : AttrDef<FIRRTLDialect, "ParamDecl"> {
  let summary = "Module or instance parameter definition";
  let description = [{
    An attribute describing a module parameter, or instance parameter
    specification.
  }];

  /// The value of the attribute - in a module, this is the default
  /// value (and may be missing).  In an instance, this is a required field that
  /// specifies the value being passed.  The verilog emitter omits printing the
  /// parameter for an instance when the applied value and the default value are
  /// the same.
  let parameters = (ins "::mlir::StringAttr":$name,
                        "::mlir::TypeAttr":$type,
                        "::mlir::Attribute":$value);
  let mnemonic = "param.decl";

  let hasCustomAssemblyFormat = 1;

  let builders = [
    AttrBuilderWithInferredContext<(ins "::mlir::StringAttr":$name,
                                         "::mlir::Type":$type),
      "auto *context = type.getContext();\n"
      "return $_get(context, name, TypeAttr::get(type), Attribute());">,
    AttrBuilderWithInferredContext<(ins "::mlir::StringRef":$name,
                                         "::mlir::Type":$type),
      "return get(StringAttr::get(type.getContext(), name), type);">,

    AttrBuilderWithInferredContext<(ins "::mlir::StringAttr":$name,
                                        "::mlir::Attribute":$value),
      "auto *context = value.getContext();\n"
      "return $_get(context, name, TypeAttr::get(value.getType()), value);">,
    AttrBuilderWithInferredContext<(ins "::mlir::StringRef":$name,
                                        "::mlir::Attribute":$value),
      "return get(StringAttr::get(value.getContext(), name), value);">
  ];

  let extraClassDeclaration = [{
    static ParamDeclAttr getWithName(ParamDeclAttr param,
                                     ::mlir::StringAttr name) {
      return get(param.getContext(), name, param.getType(), param.getValue());
    }
  }];
}

/// An array of ParamDeclAttr's that may or may not have a 'value' specified,
/// to be used on hw.module or hw.instance.  The hw.instance verifier further
/// ensures that all the values are specified.
def ParamDeclArrayAttr
  : TypedArrayAttrBase<ParamDeclAttr, "parameter array attribute">;

/// An attribute describing a kind of a name which defines name preservation
/// semantics. We preserve names tagged with `InterestingName`.
def DroppableName: I32EnumAttrCase<"DroppableName", 0, "droppable_name">;
def InterestingName: I32EnumAttrCase<"InterestingName", 1, "interesting_name">;

def NameKindEnumImpl: I32EnumAttr<"NameKindEnum", "name kind",
              [DroppableName, InterestingName]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::circt::firrtl";
}

def NameKindAttr: EnumAttr<FIRRTLDialect, NameKindEnumImpl, "name_kind">;

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLATTRIBUTES_TD
