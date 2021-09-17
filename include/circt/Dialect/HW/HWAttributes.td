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
