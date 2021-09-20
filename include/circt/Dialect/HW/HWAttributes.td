//===- HWAttributes.td - Attributes for HW dialect ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines HW dialect specific attributes.
//
//===----------------------------------------------------------------------===//

/// An attribute to indicate the output file an operation should be emitted to.
def OutputFileAttr : StructAttr<"OutputFileAttr", HWDialect, [
  StructFieldAttr<"directory", OptionalAttr<StrAttr>>,
  StructFieldAttr<"name", OptionalAttr<StrAttr>>,
  StructFieldAttr<"exclude_from_filelist",
    DefaultValuedAttr<BoolAttr, "false">>,
  StructFieldAttr<"exclude_replicated_ops",
    DefaultValuedAttr<BoolAttr, "true">>,
]>;

/// An attribute describing a module parameter, or instance parameter
/// specification.
def ParameterAttr : AttrDef<HWDialect, "Parameter"> {
  let summary = "module or instance parameter definition";

  /// The value of the attribute - in a module, this is the default
  /// value (and may be missing).  In an instance, this is a required field that
  /// specifies the value being passed.  The verilog emitter omits printing the
  /// parameter for an instance when the applied value and the default value are
  /// the same.
  let parameters = (ins "::mlir::StringAttr":$name,
                        "::mlir::TypeAttr":$type,
                        "::mlir::Attribute":$value);
  let mnemonic = "parameter";

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
    static ParameterAttr getWithName(ParameterAttr param,
                                     ::mlir::StringAttr name) {
      return get(param.getContext(), name, param.getType(), param.getValue());
    }
  }];
}

/// An array of ParameterAttr's that may or may not have a 'value' specified,
/// to be used on hw.module or hw.instance.  The hw.instance verifier further
/// ensures that all the values are specified.
def ParameterArrayAttr
  : TypedArrayAttrBase<ParameterAttr, "parameter array attribute">;

/// This attribute models a reference to a named parameter within a module body.
/// The type of the ParameterRef must always be the same as the type of the
/// parameter being referenced.
def ParameterRefAttr : AttrDef<HWDialect, "ParameterRef"> {
  let summary = "Is a reference to a parameter value.";
  let parameters = (ins "::mlir::StringAttr":$name,
                        AttributeSelfTypeParameter<"">:$type);
  let mnemonic = "parameter.ref";
}

def VerbatimParameterValueAttr : AttrDef<HWDialect, "VerbatimParameterValue"> {
  let summary =
    "Represents text to emit directly to SystemVerilog for a parameter";
  let parameters = (ins "::mlir::StringAttr":$value,
                        AttributeSelfTypeParameter<"">:$type);
  let mnemonic = "verbatim.parameter.value";
}
