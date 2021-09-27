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
def OutputFileAttr : AttrDef<HWDialect, "OutputFile"> {
  let summary = "Ouput file attribute";
  let description = [{
    This attribute represents an output file for something which will be
    printed. The `filename` string is the file to be output to. If `filename`
    ends in a `/` it is considered an output directory.

    When ExportVerilog runs, one of the files produced is a list of all other
    files which are produced. The flag `excludeFromFileList` controls if this
    file should be included in this list. If any `OutputFileAttr` referring to
    the same file sets this to `true`, it will be included in the file list.
    This option defaults to `false`.

    For each file emitted by the verilog emitter, certain prelude output will
    be included before the main content. The flag `includeReplicatedOps` can
    be used to disable the addition of the prelude text. All `OutputFileAttr`s
    referring to the same file must use a consistent setting for this value.
    This option defaults to `true`.

    Examples:
    ```mlir
      #hw.ouput_file<"/home/tester/t.sv">
      #hw.ouput_file<"t.sv", excludeFromFileList, includeReplicatedOps>
    ```
  }];
  let mnemonic = "output_file";
  let parameters = (ins "::mlir::StringAttr":$filename,
                        "::mlir::BoolAttr":$excludeFromFilelist,
                        "::mlir::BoolAttr":$includeReplicatedOps);
  let builders = [
    AttrBuilderWithInferredContext<(ins
                                   "::mlir::StringAttr":$filename,
                                   "::mlir::BoolAttr":$excludeFromFileList,
                                   "::mlir::BoolAttr":$includeReplicatedOps), [{
      return get(filename.getContext(), filename, excludeFromFileList,
                 includeReplicatedOps);
    }]>,
  ];

  let extraClassDeclaration = [{
    /// Get an OutputFileAttr from a string filename, canonicalizing the
    /// filename.
    static OutputFileAttr getFromFilename(::mlir::MLIRContext *context,
                                          const ::mlir::Twine &filename,
                                          bool excludeFromFileList = false,
                                          bool includeReplicatedOps = false);

    /// Get an OutputFileAttr from a string filename, resolving it relative to
    /// `directory`. If `filename` is an absolute path, the given `directory`
    /// will not be used.
    static OutputFileAttr getFromDirectoryAndFilename(
                                          ::mlir::MLIRContext *context,
                                          const ::mlir::Twine &directory,
                                          const ::mlir::Twine &filename,
                                          bool excludeFromFileList = false,
                                          bool includeReplicatedOps = false);

    /// Get an OutputFileAttr from a string directory name. The name will have
    /// a trailing `/` added if it is not there, ensuring that this will be
    /// an output directory.
    static OutputFileAttr getAsDirectory(::mlir::MLIRContext *context,
                                         const ::mlir::Twine &directory,
                                         bool excludeFromFileList = false,
                                         bool includeReplicatedOps = false);

    /// Returns true if this a directory.
    bool isDirectory();
  }];
}

/// An attribute describing a module parameter, or instance parameter
/// specification.
def ParamDeclAttr : AttrDef<HWDialect, "ParamDecl"> {
  let summary = "module or instance parameter definition";

  /// The value of the attribute - in a module, this is the default
  /// value (and may be missing).  In an instance, this is a required field that
  /// specifies the value being passed.  The verilog emitter omits printing the
  /// parameter for an instance when the applied value and the default value are
  /// the same.
  let parameters = (ins "::mlir::StringAttr":$name,
                        "::mlir::TypeAttr":$type,
                        "::mlir::Attribute":$value);
  let mnemonic = "param.decl";

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

/// This attribute models a reference to a named parameter within a module body.
/// The type of the ParamDeclRefAttr must always be the same as the type of the
/// parameter being referenced.
def ParamDeclRefAttr : AttrDef<HWDialect, "ParamDeclRef"> {
  let summary = "Is a reference to a parameter value.";
  let parameters = (ins "::mlir::StringAttr":$name,
                        AttributeSelfTypeParameter<"">:$type);
  let mnemonic = "param.decl.ref";
}

def ParamVerbatimAttr : AttrDef<HWDialect, "ParamVerbatim"> {
  let summary =
    "Represents text to emit directly to SystemVerilog for a parameter";
  let parameters = (ins "::mlir::StringAttr":$value,
                        AttributeSelfTypeParameter<"">:$type);
  let mnemonic = "param.verbatim";
}


let cppNamespace = "circt::hw" in {
def PBO_Add : I32EnumAttrCase<"Add", 0, "add">;
def PBO_Mul : I32EnumAttrCase<"Mul", 1, "mul">;
def PBOAttr : I32EnumAttr<"PBO", "Parameter Binary Operation Code",
                          [PBO_Add, PBO_Mul]>;
}

def ParamBinaryAttr : AttrDef<HWDialect, "ParamBinary"> {
  let summary = "Binary operation combining two parameter expressions";
  let parameters = (ins "PBO":$opcode,
                        "::mlir::Attribute":$lhs, "::mlir::Attribute":$rhs,
                        AttributeSelfTypeParameter<"">:$type);
  let mnemonic = "param.binary";

  let builders = [
    AttrBuilderWithInferredContext<(ins
                                   "PBO":$opcode,
                                   "::mlir::Attribute":$lhs,
                                   "::mlir::Attribute":$rhs), [{
      return get(lhs.getContext(), opcode, lhs, rhs, lhs.getType());
    }]>,
  ];
}

let cppNamespace = "circt::hw" in {
def WUW_Undefined : I32EnumAttrCase<"Undefined", 0>;
def WUW_PortOrder : I32EnumAttrCase<"PortOrder", 1>;
def WUWAttr : I32EnumAttr<"WUW", "Write Under Write Behavior",
                          [WUW_Undefined, WUW_PortOrder]>;
}
