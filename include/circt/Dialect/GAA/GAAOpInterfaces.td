//===- GAAOpInterfaces.td - GAA Dialect OpInterfaces -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the GAA dialect operator interfaces.
//
//===----------------------------------------------------------------------===//
include "mlir/IR/OpBase.td"

def GAAModuleLike : OpInterface<"GAAModuleLike"> {
  let cppNamespace = "::circt::gaa";
  let description = "Provide base module interface.";
  let methods = [
    //===------------------------------------------------------------------===//
    // Module Name
    //===------------------------------------------------------------------===//

    InterfaceMethod<"Get the module name",
    "StringRef", "moduleName", (ins),
    /*methodBody=*/[{ return $_op.getName(); }]>,

    InterfaceMethod<"Get the module name",
    "StringAttr", "moduleNameAttr", (ins),
    /*methodBody=*/[{ return $_op.getNameAttr(); }]>,

    //===------------------------------------------------------------------===//
    // Instance
    //===------------------------------------------------------------------===//
    // InterfaceMethod<"Get instances for this Module.",
    // "llvm::SmallVector<StringAttr>", "instances", (ins),
    // /*methodBody=*/[{ return $_op.getInstances(); }]>,
  ];
  let verify = [{}];
}

// BindMethod and Method
def GAAMethodLike : OpInterface<"GAAMethodLike"> {
  let cppNamespace = "::circt::gaa";
  let description = "Provide method interface.";
}

// BindValue and Value
def GAAValueLike : OpInterface<"GAAValueLike"> {
  let cppNamespace = "::circt::gaa";
  let description = "Provide value interface.";
}

def GAARuleLike : OpInterface<"GAARuleLike"> {
  let cppNamespace = "::circt::gaa";
  let description = "Provide rule interface.";
  let methods = [
    //===------------------------------------------------------------------===//
    // Symbol Name Handle
    //===------------------------------------------------------------------===//

    InterfaceMethod<"Get the rule name",
    "StringRef", "ruleName", (ins),
    /*methodBody=*/[{ return $_op.getName(); }]>,

    InterfaceMethod<"Get the rule name",
    "StringAttr", "ruleNameAttr", (ins),
    /*methodBody=*/[{ return $_op.getNameAttr(); }]>,
  ];
  let verify = [{}];
}

def GAAFunctionLike : OpInterface<"GAAFunctionLike"> {
  let cppNamespace = "::circt::gaa";
  let description = "Provide method and value interface to modules.";
  let methods = [
    //===------------------------------------------------------------------===//
    // Symbol Name Handle
    //===------------------------------------------------------------------===//

    InterfaceMethod<"Get the function name",
    "StringRef", "functionName", (ins),
    /*methodBody=*/[{ return $_op.getName(); }]>,

    InterfaceMethod<"Get the function name",
    "StringAttr", "functionNameAttr", (ins),
    /*methodBody=*/[{ return $_op.getNameAttr(); }]>,
  ];

}

def GAACallLike : OpInterface<"GAACallLike"> {
  let cppNamespace = "::circt::gaa";
  let description = "Provide functions to call method and value";
  let methods = [
    //===------------------------------------------------------------------===//
    // Instance
    //===------------------------------------------------------------------===//
    InterfaceMethod<"Which instance is called.",
    "SymbolRefAttr", "instanceName", (ins),
    [{}],
    /*defaultImplementation=*/[{
      return $_op->template
        getAttrOfType<SymbolRefAttr>("instanceName");
    }]
    >,

    //===------------------------------------------------------------------===//
    // Methods
    //===------------------------------------------------------------------===//
    InterfaceMethod<"Which FunctionLikeOp is called.",
    "SymbolRefAttr", "functionName", (ins),
    [{}],
    /*defaultImplementation=*/[{
      return $_op->template
        getAttrOfType<SymbolRefAttr>("functionName");
    }]
    >,
  ];
}