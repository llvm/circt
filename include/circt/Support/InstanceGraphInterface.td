//===- InstanceGraphInterface.td - Interface for instance graphs --------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains interfaces and other utilities for interacting with the
// generic CIRCT instance graph.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_INSTANCEGRAPH_INSTANCEGRAPHINTERFACE_TD
#define CIRCT_SUPPORT_INSTANCEGRAPH_INSTANCEGRAPHINTERFACE_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

def InstanceGraphInstanceOpInterface : OpInterface<"InstanceOpInterface"> {
  let description = [{
    This interface provides hooks for an instance-like operation.
  }];
  let cppNamespace = "::circt::igraph";

  let methods = [
    InterfaceMethod<"Get the name of the instance",
    "::llvm::StringRef", "getInstanceName", (ins)>,

    InterfaceMethod<"Get the name of the instance",
    "::mlir::StringAttr", "getInstanceNameAttr", (ins)>,

    InterfaceMethod<"Get the name of all the possibly instantiated modules",
    "::llvm::SmallVector<::llvm::StringRef, 1>", "getReferencedModuleNames", (ins),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{
      ::llvm::SmallVector<::llvm::StringRef, 1> targetNames;
      for (::mlir::Attribute targetNameAttr : $_op.getReferencedModuleNamesAttr())
        targetNames.push_back(
            targetNameAttr.cast<::mlir::StringAttr>().getValue());
      return targetNames;
    }]>,

    InterfaceMethod<"Get the names of the instantiated modules",
    "::mlir::ArrayAttr", "getReferencedModuleNamesAttr", (ins),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{
      auto attr = $_op.getReferencedModuleNameAttr();
      return ::mlir::ArrayAttr::get(attr.getContext(), {attr});
    }]>,
  ];
}

def InstanceGraphModuleOpInterface : OpInterface<"ModuleOpInterface"> {
  let description = [{
    This interface provides hooks for a module-like operation.
  }];
  let cppNamespace = "::circt::igraph";

  let methods = [
    InterfaceMethod<"Get the module name",
    "::llvm::StringRef", "getModuleName", (ins),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{ return $_op.getModuleNameAttr().getValue(); }]>,

    InterfaceMethod<"Get the module name",
    "::mlir::StringAttr", "getModuleNameAttr", (ins),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{ return $_op.getNameAttr(); }]>,
  ];

}

#endif // CIRCT_SUPPORT_INSTANCEGRAPH_INSTANCEGRAPHINTERFACE_TD
