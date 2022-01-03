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

def InnerRefAttr : AttrDef<HWDialect, "InnerRef"> {
  let summary = "Refer to a name inside a module";
  let description = [{
    This works like a symbol reference, but to a name inside a module.
  }];
  let mnemonic = "innerNameRef";
  let parameters = (ins "::mlir::StringAttr":$module, "::mlir::StringAttr":$name,
                        "unsigned":$fieldID);
  let builders = [
    AttrBuilderWithInferredContext<(ins "::mlir::StringAttr":$module,
                                        "::mlir::StringAttr":$name,
                                        "unsigned":$fieldID),[{
      return get(module.getContext(), module, name, fieldID);
    }]>,
    AttrBuilderWithInferredContext<(ins "::mlir::StringAttr":$module,
                                        "::mlir::StringAttr":$name),[{
      return get(module.getContext(), module, name, 0);
    }]>,
  ];
  let extraClassDeclaration = [{
    /// Get the InnerRefAttr for an operation and add the sym on it.
    static InnerRefAttr getFromOperation(mlir::Operation *op,
                  mlir::StringAttr symName, mlir::StringAttr moduleName,
                  unsigned fieldID = 0);
    static InnerRefAttr get(mlir::MLIRContext* ctx,
                  mlir::StringAttr symName, mlir::StringAttr moduleName){
      return get(ctx, symName, moduleName, 0);
    }
  }];
}

def GlobalRefAttr : AttrDef<HWDialect, "GlobalRef"> {
  let summary = "Refer to a non-local symbol";
  let description = [{
    This works like a symbol reference, but to a global symbol with a possible
    unique instance path.
  }];
  let mnemonic = "globalNameRef";
  let parameters = (ins "::mlir::FlatSymbolRefAttr":$glblSym);
}
