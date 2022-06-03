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

def HierarchicalPathLike : OpInterface<"HierarchicalPathLike"> {
  let cppNamespace = "circt::hw";
  let description = [{
    Represents a path through the hierarchy.
  }];
  let methods = [
    InterfaceMethod<
      "Get the path",
      "::mlir::ArrayAttr",
      "namepath",
      (ins)>,

    InterfaceMethod<
      "Drop the module from the namepath. If its a InnerNameRef, then drop the Module-Instance pair, else drop the final module from the namepath. Return true if any update is made.",
      "bool",
      "dropModule",
      (ins
        "::mlir::StringAttr":$moduleToDrop)>,

    InterfaceMethod<
      "Inline the module in the namepath. Update the symbol name for the inlined module instance, by prepending the symbol name of the instance at which the inlining was done. Return true if any update is made.",
      "bool",
      "inlineModule",
      (ins
        "::mlir::StringAttr":$moduleToDrop)>,

    InterfaceMethod<
      "Replace the oldMod module with newMod module in the namepath of the NLA. Return true if any update is made.",
      "bool",
      "updateModule",
      (ins
        "::mlir::StringAttr":$oldMod,
        "::mlir::StringAttr":$newMod)>,

    InterfaceMethod<
      "Replace the oldMod module with newMod module in the namepath of the NLA. Since the module is being updated, the symbols inside the module should also be renamed. Use the rename Map to update the corresponding inner_sym names in the namepath. Return true if any update is made.",
      "bool",
      "updateModuleAndInnerRef",
      (ins
        "::mlir::StringAttr":$oldMod,
        "::mlir::StringAttr":$newMod,
        "const llvm::DenseMap<mlir::StringAttr, mlir::StringAttr> &":$innerSymRenameMap)>,

    InterfaceMethod<
      "Truncate the namepath for this NLA, at atMod module. If includeMod is false, drop atMod and beyond, else include it and drop everything after it. Return true if any update is made.",
      "bool",
      "truncateAtModule",
      (ins
        "::mlir::StringAttr":$atMod,
        "bool":$includeMod)>,

    InterfaceMethod<
      "Return just the module part of the namepath at a specific index.",
      "::mlir::StringAttr",
      "modPart",
      (ins
        "unsigned":$i)>,

    InterfaceMethod<
      "Return the root module.",
      "::mlir::StringAttr",
      "root",
      (ins)>,

    InterfaceMethod<
      "Return just the reference part of the namepath at a specific index. This will return an empty attribute if this is the leaf and the leaf is a module.",
      "::mlir::StringAttr",
      "refPart",
      (ins
        "unsigned":$i)>,

    InterfaceMethod<
      "Return the leaf reference.  This returns an empty attribute if the leaf reference is a module.",
      "::mlir::StringAttr",
      "ref",
      (ins)>,

    InterfaceMethod<
      "Return the leaf Module.",
      "::mlir::StringAttr",
      "leafMod",
      (ins)>,

    InterfaceMethod<
      "Returns true, if the NLA path contains the module.",
      "bool",
      "hasModule",
      (ins
        "::mlir::StringAttr":$modName)>,

    InterfaceMethod<
      "Returns true, if the NLA path contains the InnerSym {modName, symName}.",
      "bool",
      "hasInnerSym",
      (ins "::mlir::StringAttr":$modName, "::mlir::StringAttr":$symName)>,

    InterfaceMethod<
      "Returns true if this NLA targets a module or instance of a module (as opposed to an instance's port or something inside an instance).",
      "bool",
      "isModule",
      (ins)>,

    InterfaceMethod<
      "Returns true if this NLA targets something inside a module (as opposed to a module or an instance of a module)",
      "bool",
      "isComponent",
      (ins)>,
    ];
}


#endif
