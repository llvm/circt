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

    InterfaceMethod<"Get the number of ports",
    "size_t", "getNumPorts">,

    InterfaceMethod<"Get a port symbol attribute",
    "::circt::hw::InnerSymAttr", "getPortSymbolAttr", (ins "size_t":$portIndex)>
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

    /// Appends output ports to the module with the specified names and rewrites
    /// the output op to return the associated values.
    InterfaceMethod<"Append output values to this module",
    "void", "appendOutputs", (ins
      "ArrayRef<std::pair<StringAttr, Value>>":$outputs)>
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

    InterfaceMethod<"Get the name of the instance",
    "::mlir::StringAttr", "instanceNameAttr", (ins),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{ return $_op.getInstanceNameAttr(); }]>,

    InterfaceMethod<"Get the name of the instantiated module",
    "::llvm::StringRef", "referencedModuleName", (ins),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{ return $_op.getModuleName(); }]>,

    InterfaceMethod<"Get the name of the instantiated module",
    "::mlir::StringAttr", "referencedModuleNameAttr", (ins),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{ return $_op.getModuleNameAttr().getAttr(); }]>,

    InterfaceMethod<"Get the referenced module",
    "::mlir::Operation *", "getReferencedModule", (ins),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{}]>,
  ];
}

def InnerRefNamespace : NativeOpTrait<"InnerRefNamespace">;

def InnerSymbol : OpInterface<"InnerSymbolOpInterface"> {
  let description = [{
    This interface describes an operation that may define an
    `inner_sym`.  An `inner_sym` operation resides
    in arbitrarily-nested regions of a region that defines a
    `InnerSymbolTable`.
    Inner Symbols are different from normal symbols due to
    MLIR symbol table resolution rules.  Specifically normal
    symbols are resolved by first going up to the closest
    parent symbol table and resolving from there (recursing
    down for complex symbol paths).  In HW and SV, modules
    define a symbol in a circuit or std.module symbol table.
    For instances to be able to resolve the modules they
    instantiate, the symbol use in an instance must resolve
    in the top-level symbol table.  If a module were a
    symbol table, instances resolving a symbol would start from
    their own module, never seeing other modules (since
    resolution would start in the parent module of the
    instance and be unable to go to the global scope).
    The second problem arises from nesting.  Symbols defining
    ops must be immediate children of a symbol table.  HW
    and SV operations which define a inner_sym are grandchildren,
    at least, of a symbol table and may be much further nested.
    Lastly, ports need to define inner_sym, something not allowed
    by normal symbols.

    Any operation implementing an InnerSymbol may have the inner symbol be
    optional and all methods should be robuse to the attribute not being
    defined.
  }];

  let cppNamespace = "::circt::hw";
  let methods = [
    InterfaceMethod<"Returns the name of this inner symbol.",
      "::mlir::StringAttr", "getInnerNameAttr", (ins), [{}],
      /*defaultImplementation=*/[{
        if (auto attr =
              this->getOperation()->template getAttrOfType<hw::InnerSymAttr>(
              circt::hw::InnerSymbolTable::getInnerSymbolAttrName()))
          return attr.getSymName();
        return {};
      }]
    >,
    InterfaceMethod<"Returns the name of this inner symbol.",
      "::std::optional<::mlir::StringRef>", "getInnerName", (ins), [{}],
      /*defaultImplementation=*/[{
        auto attr = this->getInnerNameAttr();
        return attr ? ::std::optional<StringRef>(attr.getValue()) : ::std::nullopt;
      }]
    >,
    InterfaceMethod<"Sets the name of this inner symbol.",
      "void", "setInnerSymbol", (ins "::mlir::StringAttr":$name), [{}],
      /*defaultImplementation=*/[{
        this->getOperation()->setAttr(
            InnerSymbolTable::getInnerSymbolAttrName(), hw::InnerSymAttr::get(name));
      }]
    >,
    InterfaceMethod<"Returns an InnerRef to this operation.  Must have inner symbol.",
      "::circt::hw::InnerRefAttr", "getInnerRef", (ins), [{}],
      /*defaultImplementation=*/[{
        auto *op = this->getOperation();
        return hw::InnerRefAttr::get(
            SymbolTable::getSymbolName(
                op->template getParentWithTrait<OpTrait::InnerSymbolTable>()),
            InnerSymbolTable::getInnerSymbol(op));
      }]
    >,
    InterfaceMethod<"Returns the InnerSymAttr.",
      "::circt::hw::InnerSymAttr", "getInnerSymAttr", (ins), [{}],
      /*defaultImplementation=*/[{
        return this->getOperation()->template getAttrOfType<hw::InnerSymAttr>(
              circt::hw::InnerSymbolTable::getInnerSymbolAttrName());
      }]
    >,
  ];

  let verify = [{
    return verifyInnerSymAttr(op);
  }];
}

def InnerSymbolTable : NativeOpTrait<"InnerSymbolTable">;

def InnerRefUserOpInterface : OpInterface<"InnerRefUserOpInterface"> {
  let description = [{
    This interface describes an operation that may use a `InnerRef`. This
    interface allows for users of inner symbols to hook into verification and
    other inner symbol related utilities that are either costly or otherwise
    disallowed within a traditional operation.
  }];
  let cppNamespace = "::circt::hw";

  let methods = [
    InterfaceMethod<"Verify the inner ref uses held by this operation.",
      "::mlir::LogicalResult", "verifyInnerRefs",
      (ins "::circt::hw::InnerRefNamespace&":$ns)
    >,
  ];
}

#endif
