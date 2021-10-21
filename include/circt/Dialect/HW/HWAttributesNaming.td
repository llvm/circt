//===- HWAttributesNaming.td - Attributes for HW dialect ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines HW dialect attributes used in other dialects.
//
//===----------------------------------------------------------------------===//


def InnerDefAttr : AttrDef<HWDialect, "InnerDef"> {
  let summary = "Marks an entity visible inside a module";
  let description = [{
    This works like a symbol, but inside a module and can be applied to non-ops (ports).
  }];
  let mnemonic = "innerName";
  let parameters = (ins "::mlir::StringAttr":$inner_sym);
}

def InnerRefAttr : AttrDef<HWDialect, "InnerRef"> {
  let summary = "Refer to a name inside a module";
  let description = [{
    This works like a symbol reference, but to a name inside a module.
  }];
  let mnemonic = "innerNameRef";
  let parameters = (ins "::mlir::StringAttr":$module, "::mlir::StringAttr":$name);
  let builders = [
    AttrBuilderWithInferredContext<(ins "::mlir::StringAttr":$module, "::mlir::StringAttr":$name),[{
      return get(module.getContext(), module, name);
    }]>,
  ];
}
