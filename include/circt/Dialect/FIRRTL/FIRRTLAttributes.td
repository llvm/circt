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

include "FIRRTLDialect.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

//===----------------------------------------------------------------------===//
// FIRRTL Annotations Definition
//===----------------------------------------------------------------------===//

def AnnotationArrayAttr
  : TypedArrayAttrBase<DictionaryAttr, "Annotation array attribute">;

def PortAnnotationsAttr : ArrayAttrBase<
    And<[
      // Guarantee this is an ArrayAttr first
      CPred<"$_self.isa<::mlir::ArrayAttr>()">,
      // Guarantee all elements are an array or a dictionary.
      CPred<"::llvm::all_of($_self.cast<::mlir::ArrayAttr>(), "
            "[&](::mlir::Attribute attr) { return attr.isa<"
            "::mlir::ArrayAttr,"
            "::mlir::DictionaryAttr>();})">]>,
    "Port annotations attribute"> {
  let constBuilderCall = "$_builder.getArrayAttr($0)";
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

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLATTRIBUTES_TD
