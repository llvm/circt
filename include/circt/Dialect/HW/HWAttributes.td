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
def ParameterAttr : StructAttr<"ParameterAttr", HWDialect, [
  /// This is the name of the parameter.
  StructFieldAttr<"name", StrAttr>,
  /// This is the MLIR type for it.
  StructFieldAttr<"type", TypeAttr>,
  /// This is the value of the attribute - in a module, this is the default
  /// value (and may be missing).  In an instance, this is a required field that
  /// specifies the value being passed.  The verilog emitter omits printing the
  /// parameter for an instance when the applied value and the default value are
  /// the same.
  StructFieldAttr<"value", OptionalAttr<AnyAttr>>
]>;

/// An array of ParameterAttr's that may or may not have a 'value' specified,
/// to be used on hw.module or hw.instance.  The hw.instance verifier further
/// ensures that all the values are specified.
def ParameterArrayAttr
  : TypedArrayAttrBase<ParameterAttr, "parameter array attribute">;
