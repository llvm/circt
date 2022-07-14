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
  let description = "Provide common module information.";

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

def HWMutableModuleLike : OpInterface<"HWMutableModuleLike"> {
  let cppNamespace = "circt::hw";
  let description = "Provide methods to mutate a module.";

  let methods = [
    InterfaceMethod<"Return the number of inputs to this module",
    "unsigned", "getNumInputs", (ins),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{
      return $_op->template getAttrOfType<ArrayAttr>("argNames").size();
    }]>,

    InterfaceMethod<"Return the number of outputs from this module",
    "unsigned", "getNumOutputs", (ins),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{
      return $_op->template getAttrOfType<ArrayAttr>("resultNames").size();
    }]>,

    InterfaceMethod<"Return the names of the inputs this module",
    "mlir::ArrayAttr", "getArgNames", (ins),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{
      return $_op->template getAttrOfType<ArrayAttr>("argNames");
    }]>,

    InterfaceMethod<"Return the names of the outputs this module",
    "mlir::ArrayAttr", "getResultNames", (ins),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{
      return $_op->template getAttrOfType<ArrayAttr>("resultNames");
    }]>,

    /// Insert and remove input and output ports of this module. Does not modify
    /// the block arguments of the module body. The insertion and removal
    /// indices must be in ascending order. The indices refer to the port
    /// positions before any insertion or removal occurs. Ports inserted at the
    /// same index will appear in the module in the same order as they were
    /// listed in the insertion arrays.
    InterfaceMethod<"Insert and remove input and output ports",
    "void", "modifyPorts", (ins
      "ArrayRef<std::pair<unsigned, circt::hw::PortInfo>>":$insertInputs,
      "ArrayRef<std::pair<unsigned, circt::hw::PortInfo>>":$insertOutputs,
      "ArrayRef<unsigned>":$eraseInputs, "ArrayRef<unsigned>":$eraseOutputs),
    /*methodBody=*/[{
      $_op.modifyPorts(insertInputs, insertOutputs, eraseInputs, eraseOutputs);
    }]>,

    /// Insert ports into the module. Does not modify the block arguments of the
    /// module body.
    InterfaceMethod<"Insert ports into this module",
    "void", "insertPorts", (ins
      "ArrayRef<std::pair<unsigned, circt::hw::PortInfo>>":$insertInputs,
      "ArrayRef<std::pair<unsigned, circt::hw::PortInfo>>":$insertOutputs),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{
      $_op.modifyPorts(insertInputs, insertOutputs, {}, {});
    }]>,

    /// Erase ports from the module. Does not modify the block arguments of the
    /// module body.
    InterfaceMethod<"Erase ports from this module",
    "void", "erasePorts", (ins
      "ArrayRef<unsigned>":$eraseInputs,
      "ArrayRef<unsigned>":$eraseOutputs),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{
      $_op.modifyPorts({}, {}, eraseInputs, eraseOutputs);
    }]>,
  ];
}


def HWInstanceLike : OpInterface<"HWInstanceLike"> {
  let cppNamespace = "circt::hw";
  let description = "Provide common  module information.";

  let methods = [
    InterfaceMethod<"Get the name of the instance",
    "::llvm::StringRef", "instanceName", (ins),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{ return $_op.getInstanceName(); }]>,

    InterfaceMethod<"Get the name of the instantiated module",
    "::llvm::StringRef", "referencedModuleName", (ins),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{ return $_op.getModuleName(); }]>,

    InterfaceMethod<"Get the name of the instantiated module",
    "::mlir::StringAttr", "referencedModuleNameAttr", (ins),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{ return $_op.getModuleNameAttr().getAttr(); }]>,
  ];
}

#endif
