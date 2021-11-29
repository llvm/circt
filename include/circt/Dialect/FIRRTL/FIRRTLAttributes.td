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
    ""> {
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
    ""> {
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

  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type),
                                   "return $_get(type.getContext(), type);">
  ];
}

//===----------------------------------------------------------------------===//
// Grand Central-related Annotations
//===----------------------------------------------------------------------===//

class AugmentedType<string name> : AttrDef<FIRRTLDialect, name> {
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
  let extraClassDeclaration = defaultClassDeclaration # hasElements;
}

def AugmentedGroundType : AugmentedType<"AugmentedGroundType"> {
  let extraClassDeclaration = hasID # hasName;
}

def AugmentedStringType : AugmentedType<"AugmentedStringType"> {
  let extraClassDeclaration = hasName;
}
def AugmentedBooleanType : AugmentedType<"AugmentedBooleanType"> {
  let extraClassDeclaration = hasName;
}
def AugmentedIntegerType : AugmentedType<"AugmentedIntegerType"> {
  let extraClassDeclaration = hasName;
}
def AugmentedDoubleType : AugmentedType<"AugmentedDoubleType"> {
  let extraClassDeclaration = hasName;
}
def AugmentedLiteralType : AugmentedType<"AugmentedLiteralType"> {
  let extraClassDeclaration = hasName;
}
def AugmentedDeletedType : AugmentedType<"AugmentedDeletedType"> {
  let extraClassDeclaration = hasName;
}

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLATTRIBUTES_TD
