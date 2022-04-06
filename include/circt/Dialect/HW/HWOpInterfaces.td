//===- HWOpInterfaces.td - Operation Interfaces ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the HW operation interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_HWOPINTERFACES
#define CIRCT_DIALECT_HW_HWOPINTERFACES

include "mlir/IR/OpBase.td"

def HWModuleLike : OpInterface<"HWModuleLike"> {
  let cppNamespace = "circt::hw";
  let description = "Provide common  module information.";

  let methods = [
    InterfaceMethod<"Check whether the module is publicly visible",
    "bool", "isPublic", (ins),
    /*methodBody=*/[{
      return ::mlir::SymbolTable::getSymbolVisibility($_op) ==
             ::mlir::SymbolTable::Visibility::Public;
    }]>,

    InterfaceMethod<"Get the module name",
    "::llvm::StringRef", "moduleName", (ins),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{ return $_op.getName(); }]>,

    InterfaceMethod<"Get the module name",
    "::mlir::StringAttr", "moduleNameAttr", (ins),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{ return $_op.getNameAttr(); }]>,
  ];

  let verify = [{
    static_assert(
        ConcreteOp::template hasTrait<::mlir::SymbolOpInterface::Trait>(),
        "expected operation to be a symbol");
    return success();
  }];
}


def HWInstanceLike : OpInterface<"HWInstanceLike"> {
  let cppNamespace = "circt::hw";
  let description = "Provide common  module information.";

  let methods = [
    InterfaceMethod<"Get the name of the instance",
    "::llvm::StringRef", "instanceName", (ins),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{ return $_op.instanceName(); }]>,

    InterfaceMethod<"Get the name of the instantiated module",
    "::llvm::StringRef", "referencedModuleName", (ins),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{ return $_op.moduleName(); }]>,

    InterfaceMethod<"Get the name of the instantiated module",
    "::mlir::StringAttr", "referencedModuleNameAttr", (ins),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{ return $_op.moduleNameAttr().getAttr(); }]>,
  ];
}

#endif
