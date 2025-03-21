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

include "FIRRTLEnums.td"
include "FIRRTLDialect.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "circt/Types.td"

//===----------------------------------------------------------------------===//
// FIRRTL Annotations Definition
//===----------------------------------------------------------------------===//

def AnnotationArrayAttr
  : TypedArrayAttrBase<DictionaryAttr, "Annotation array attribute">;

def PortAnnotationsAttr : ArrayAttrBase<
    And<[
      // Guarantee this is an ArrayAttr first
      CPred<"llvm::isa<::mlir::ArrayAttr>($_self)">,
      // Guarantee all elements are an array or a dictionary.
      CPred<"::llvm::all_of(cast<mlir::ArrayAttr>($_self), "
            "[&](::mlir::Attribute attr) { return isa<"
            "::mlir::ArrayAttr,"
            "::mlir::DictionaryAttr>(attr);})">]>,
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

}

def AugmentedBundleType : AugmentedType<"AugmentedBundleType"> {
  let summary = "GrandCentral AugmentedBundleType";
  let extraClassDeclaration =
    defaultClassDeclaration #
    hasID #
    hasElements #
    hasDefName # [{
      bool isRoot() { return getID() != nullptr; }
    }];
  let mnemonic = "augmentedBundle";
  let assemblyFormat = "`<` $underlying `>`";
}

def AugmentedVectorType : AugmentedType<"AugmentedVectorType"> {
  let summary = "GrandCentral AugmentedVectorType";
  let extraClassDeclaration = defaultClassDeclaration # hasElements;
  let mnemonic = "augmentedVector";
  let assemblyFormat = "`<` $underlying `>`";
}

def AugmentedGroundType : AugmentedType<"AugmentedGroundType"> {
  let summary = "GrandCentral AugmentedGroundType";
  let extraClassDeclaration = hasID # hasName;
  let mnemonic = "augmentedGround";
  let assemblyFormat = "`<` $underlying `>`";
}

def ParamDeclAttr : AttrDef<FIRRTLDialect, "ParamDecl", [TypedAttrInterface]> {
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
                        AttributeSelfTypeParameter<"">:$type,
                        "::mlir::Attribute":$value);
  let mnemonic = "param.decl";

  let hasCustomAssemblyFormat = 1;

  let builders = [
    AttrBuilderWithInferredContext<(ins "::mlir::StringAttr":$name,
                                        "::mlir::Type":$type),
      "auto *context = type.getContext();\n"
      "return $_get(context, name, type, Attribute());">,
    AttrBuilderWithInferredContext<(ins "::mlir::StringRef":$name,
                                        "::mlir::Type":$type),
      "return get(StringAttr::get(type.getContext(), name), type);">,

    AttrBuilderWithInferredContext<(ins "::mlir::StringAttr":$name,
                                        "::mlir::TypedAttr":$value),
      "auto *context = value.getContext();\n"
      "return $_get(context, name, value.getType(), value);">,
    AttrBuilderWithInferredContext<(ins "::mlir::StringRef":$name,
                                        "::mlir::TypedAttr":$value),
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

/// An attribute holding information about memory initialization.
def MemoryInitAttr : AttrDef<FIRRTLDialect, "MemoryInit"> {
  let mnemonic = "meminit";
  let summary = "Information about the initial state of a memory";
  let description = [{
    This attribute captures information about the external initialization of a
    memory.  This is the FIRRTL Dialect representation of both
    "firrtl.annotations.LoadMemoryFromFile" and
    "firrtl.annotations.MemoryFileInlineAnnotation".
  }];
  let parameters = (ins
    "::mlir::StringAttr":$filename,
    "bool":$isBinary,
    "bool":$isInline
  );
 let assemblyFormat = "`<` $filename `,` $isBinary `,` $isInline `>`";
}

/// An attribute holding internal path for ref-type ports.
def InternalPathAttr : AttrDef<FIRRTLDialect, "InternalPath"> {
  let summary = "Internal path for ref-type ports";
  let mnemonic = "internalpath";
  let parameters = (ins OptionalParameter<"::mlir::StringAttr">:$path);
  let assemblyFormat = "(`<` $path^ `>`)?";
  let builders = [
     AttrBuilderWithInferredContext<(ins "::mlir::StringAttr":$path), [{
       return $_get(path.getContext(), path);
     }]>,
     AttrBuilder<(ins "::llvm::StringRef":$path), [{
       return $_get($_ctxt, ::mlir::StringAttr::get($_ctxt, path));
     }]>,
     AttrBuilder<(ins), [{
       return $_get($_ctxt, ::mlir::StringAttr());
     }]>
  ];
}

def InternalPathArrayAttr
  : TypedArrayAttrBase<InternalPathAttr, "InternalPath array attribute">;

//===----------------------------------------------------------------------===//
// Printf attributes
//===----------------------------------------------------------------------===//

def PrintfSubstitutionEnumAttr
  : EnumAttr<FIRRTLDialect, PrintfSubstitutionEnum, "printfSubstitution">;

def PrintfSubstitutionEnumArrayAttr
  : TypedArrayAttrBase<PrintfSubstitutionEnumAttr, "Array of special printf substitutuions">;

//===----------------------------------------------------------------------===//
// Miscellaneous attributes
//===----------------------------------------------------------------------===//

def LayerArrayAttr : TypedArrayRefAttrBase<SymbolRefAttr, "an array of layers">;

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLATTRIBUTES_TD
