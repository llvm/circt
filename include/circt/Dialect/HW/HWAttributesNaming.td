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

#ifndef CIRCT_DIALECT_HW_HWATTRIBUTESNAMING
#define CIRCT_DIALECT_HW_HWATTRIBUTESNAMING

include "circt/Dialect/HW/HWDialect.td"
include "mlir/IR/AttrTypeBase.td"

def InnerRefAttr : AttrDef<HWDialect, "InnerRef"> {
  let summary = "Refer to a name inside a list of scopes";
  let description = [{
    This works like a symbol reference, but to a name inside a list of `InnerSymbolTable`'s.
  }];
  let mnemonic = "innerNameRef";
  let parameters = (ins ArrayRefParameter<"::mlir::StringAttr">:$path);

  let builders = [

    AttrBuilderWithInferredContext<(ins "::mlir::ArrayRef<::mlir::StringAttr>":$pathAttr), [{
      assert(!pathAttr.empty() && "Empty path");
      assert(pathAttr.front() && "Null path element");
      return InnerRefAttr::get(pathAttr.front().getContext(), pathAttr);
    }]>,
  
    AttrBuilderWithInferredContext<(ins "::mlir::ArrayRef<::mlir::StringAttr>":$pathAttr,
                                        "::mlir::StringAttr":$target), [{
      llvm::SmallVector<::mlir::StringAttr> fullPath(pathAttr.begin(), pathAttr.end());
      fullPath.push_back(target);
      return InnerRefAttr::get(target.getContext(), fullPath);
    }]>,

    // @mortbopet: Old InnerRef single-module compatability builders.
    AttrBuilderWithInferredContext<(ins "::mlir::StringAttr":$target), [{
      llvm::SmallVector<::mlir::StringAttr> fullPath;
      fullPath.push_back(target);
      return InnerRefAttr::get(fullPath);
    }]>,

    AttrBuilderWithInferredContext<(ins 
                                    "::mlir::StringAttr":$parent,
                                    "::mlir::StringAttr":$target), [{
      ::llvm::SmallVector<::mlir::StringAttr> fullPath{parent};
      fullPath.push_back(target);
      return InnerRefAttr::get(fullPath);
    }]>,
    AttrBuilder<(ins "::mlir::StringAttr":$parent,
                     "::mlir::StringAttr":$target), [{
      return InnerRefAttr::get(parent, target);
    }]>,
  ];

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    ::mlir::StringAttr getRoot() const {
      assert(!getPath().empty() && "Empty path");
      return getPath().front();
    }
    ::mlir::StringAttr getTarget() const {
      assert(!getPath().empty() && "Empty path");
      return getPath().back();
    }

    // Old InnerRefAttr compatability accessors
    ::mlir::StringAttr getModule() const {
      return getRoot();
    }
    ::mlir::FlatSymbolRefAttr getModuleRef() const {
      return ::mlir::FlatSymbolRefAttr::get(getRoot());
    }
  }];
}

#endif // CIRCT_DIALECT_HW_HWATTRIBUTESNAMING
