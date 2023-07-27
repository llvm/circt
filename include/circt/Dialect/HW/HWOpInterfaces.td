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

include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/OpBase.td"

def HWModuleLike : OpInterface<"HWModuleLike", [Symbol]> {
  let cppNamespace = "circt::hw";
  let description = "Provide common module information.";

  let methods = [

    InterfaceMethod<"Get the module name",
    "::llvm::StringRef", "getModuleName", (ins),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{ return $_op.getName(); }]>,

    InterfaceMethod<"Get the module name",
    "::mlir::StringAttr", "getModuleNameAttr", (ins),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{ return $_op.getNameAttr(); }]>,

    InterfaceMethod<"Get the number of ports",
    "size_t", "getNumPorts">,

    InterfaceMethod<"Get a port symbol attribute",
    "::circt::hw::InnerSymAttr", "getPortSymbolAttr", (ins "size_t":$portIndex)>,

    InterfaceMethod<"Get the module type",
    "::circt::hw::ModuleType", "getHWModuleType", (ins),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{ return ::circt::hw::detail::fnToMod($_op, getInputNames(), getOutputNames()); }]>,

    InterfaceMethod<"Return the names of the inputs to this module",
    "mlir::ArrayAttr", "getInputNames", (ins),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{
      return $_op->template getAttrOfType<ArrayAttr>("argNames");
    }]>,

    InterfaceMethod<"Return the names of the outputs this module",
    "mlir::ArrayAttr", "getOutputNames", (ins),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{
      return $_op->template getAttrOfType<ArrayAttr>("resultNames");
    }]>,

    InterfaceMethod<"Return the locations of the inputs to this module",
    "mlir::ArrayAttr", "getInputLocs", (ins),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{
      return $_op->template getAttrOfType<ArrayAttr>("argLocs");
    }]>,

    InterfaceMethod<"Return the locations of the outputs of this module",
    "mlir::ArrayAttr", "getOutputLocs", (ins),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{
      return $_op->template getAttrOfType<ArrayAttr>("resultLocs");
    }]>,

    // Kill these eventually, inherited from FunctionInterface:
    InterfaceMethod<[{
      Returns the function argument types based exclusively on
      the type (to allow for this method may be called on function
      declarations).
    }],
    "::llvm::ArrayRef<::mlir::Type>", "getArgumentTypes">,

    InterfaceMethod<[{
      Returns the function result types based exclusively on
      the type (to allow for this method may be called on function
      declarations).
    }],
    "::llvm::ArrayRef<::mlir::Type>", "getResultTypes">,

    InterfaceMethod<[{
      Get the array of argument attribute dictionaries. The method should return
      an array attribute containing only dictionary attributes equal in number
      to the number of function arguments. Alternatively, the method can return
      null to indicate that the function has no argument attributes.
    }],
    "::mlir::ArrayAttr", "getArgAttrsAttr">,

    InterfaceMethod<[{
      Get the array of result attribute dictionaries. The method should return
      an array attribute containing only dictionary attributes equal in number
      to the number of function results. Alternatively, the method can return
      null to indicate that the function has no result attributes.
    }],
    "::mlir::ArrayAttr", "getResAttrsAttr">,

    InterfaceMethod<[{
      Set the array of argument attribute dictionaries.
    }],
    "void", "setArgAttrsAttr", (ins "::mlir::ArrayAttr":$attrs)>,

    InterfaceMethod<[{
      Set the array of result attribute dictionaries.
    }],
    "void", "setResAttrsAttr", (ins "::mlir::ArrayAttr":$attrs)>,
    InterfaceMethod<[{
      Remove the array of argument attribute dictionaries. This is the same as
      setting all argument attributes to an empty dictionary. The method should
      return the removed attribute.
    }],
    "::mlir::Attribute", "removeArgAttrsAttr">,
    InterfaceMethod<[{
      Remove the array of result attribute dictionaries. This is the same as
      setting all result attributes to an empty dictionary. The method should
      return the removed attribute.
    }],
    "::mlir::Attribute", "removeResAttrsAttr">,

  ];

  let extraSharedClassDeclaration = [{
    /// Block list iterator types.
    using BlockListType = Region::BlockListType;
    using iterator = BlockListType::iterator;
    using reverse_iterator = BlockListType::reverse_iterator;

    /// Block argument iterator types.
    using BlockArgListType = Region::BlockArgListType;
    using args_iterator = BlockArgListType::iterator;

    /// Return the region containing the body of this module.
    Region &getModuleBody() { return $_op->getRegion(0); }

    // Return the number of inputs to this module
    unsigned getNumInputs() {
      return getInputNames().size();
    }

    // Return the number of outputs from this module
    unsigned getNumOutputs() {
      return getOutputNames().size();
    }

    // Return the name of an input.
    mlir::StringRef getInputName(unsigned idx) {
      return getInputNameAttr(idx).getValue();
    }

    // Return the name of an output
    mlir::StringRef getOutputName(unsigned idx) {
      return getOutputNameAttr(idx).getValue();
    }

    // Return the name of an input to this module
    mlir::StringAttr getInputNameAttr(unsigned idx) {
      return cast<StringAttr>(getInputNames()[idx]);
    }

    // Return the name of an output to this module
    mlir::StringAttr getOutputNameAttr(unsigned idx) {
      return cast<StringAttr>(getOutputNames()[idx]);
    }

    //===------------------------------------------------------------------===//
    // Argument and Result Handling
    //===------------------------------------------------------------------===//

    /// Returns the number of function arguments.
    unsigned getNumArguments_HWML() { return $_op.getArgumentTypes().size(); }

    /// Returns the number of function results.
    unsigned getNumResults_HWML() { return $_op.getResultTypes().size(); }

    /// Returns the entry block function argument at the given index.
    BlockArgument getArgument_HWML(unsigned idx) {
      return getModuleBody().getArgument(idx);
    }

    /// Support argument iteration.
    args_iterator args_begin_HWML() { return getModuleBody().args_begin(); }
    args_iterator args_end_HWML() { return getModuleBody().args_end(); }
    BlockArgListType getArguments_HWML() { return getModuleBody().getArguments(); }


    //===------------------------------------------------------------------===//
    // Argument Attributes
    //===------------------------------------------------------------------===//

    /// Return all of the attributes for the argument at 'index'.
    ArrayRef<NamedAttribute> getArgAttrs_HWML(unsigned index) {
      return ::circt::hw::HWModuleLike_impl::getArgAttrs($_op, index);
    }

    /// Return an ArrayAttr containing all argument attribute dictionaries of
    /// this function, or nullptr if no arguments have attributes.
    ArrayAttr getAllArgAttrs_HWML() { return $_op.getArgAttrsAttr(); }

    /// Return all argument attributes of this function.
    void getAllArgAttrs_HWML(SmallVectorImpl<DictionaryAttr> &result) {
      if (ArrayAttr argAttrs = getAllArgAttrs_HWML()) {
        auto argAttrRange = argAttrs.template getAsRange<DictionaryAttr>();
        result.append(argAttrRange.begin(), argAttrRange.end());
      } else {
        result.append($_op.getNumArguments_HWML(),
                      DictionaryAttr::get(this->getOperation()->getContext()));
      }
    }

    /// Set the attributes held by the argument at 'index'.
    void setArgAttrs_HWML(unsigned index, ArrayRef<NamedAttribute> attributes) {
      ::circt::hw::HWModuleLike_impl::setArgAttrs($_op, index, attributes);
    }

    /// Set the attributes held by the argument at 'index'. `attributes` may be
    /// null, in which case any existing argument attributes are removed.
    void setArgAttrs_HWML(unsigned index, DictionaryAttr attributes) {
      ::circt::hw::HWModuleLike_impl::setArgAttrs($_op, index, attributes);
    }

    /// If the an attribute exists with the specified name, change it to the new
    /// value. Otherwise, add a new attribute with the specified name/value.
    void setArgAttr_HWML(unsigned index, StringAttr name, Attribute value) {
      ::circt::hw::HWModuleLike_impl::setArgAttr($_op, index, name, value);
    }
    void setArgAttr_HWML(unsigned index, StringRef name, Attribute value) {
      setArgAttr_HWML(index,
                 StringAttr::get(this->getOperation()->getContext(), name),
                 value);
    }

    /// Returns the dictionary attribute corresponding to the argument at
    /// 'index'. If there are no argument attributes at 'index', a null
    /// attribute is returned.
    DictionaryAttr getArgAttrDict_HWML(unsigned index) {
      assert(index < $_op.getNumArguments_HWML() && "invalid argument number");
      return ::circt::hw::HWModuleLike_impl::getArgAttrDict($_op, index);
    }

    /// Remove the attribute 'name' from the argument at 'index'. Return the
    /// attribute that was erased, or nullptr if there was no attribute with
    /// such name.
    Attribute removeArgAttr(unsigned index, StringAttr name) {
      return ::circt::hw::HWModuleLike_impl::removeArgAttr($_op, index, name);
    }

    //===------------------------------------------------------------------===//
    // Result Attributes
    //===------------------------------------------------------------------===//

    /// Return all of the attributes for the result at 'index'.
    ArrayRef<NamedAttribute> getResultAttrs_HWML(unsigned index) {
      return ::circt::hw::HWModuleLike_impl::getResultAttrs($_op, index);
    }

    /// Return an ArrayAttr containing all result attribute dictionaries of this
    /// function, or nullptr if no result have attributes.
    ArrayAttr getAllResultAttrs_HWML() { return $_op.getResAttrsAttr(); }

    /// Return all result attributes of this function.
    void getAllResultAttrs_HWML(SmallVectorImpl<DictionaryAttr> &result) {
      if (ArrayAttr argAttrs = getAllResultAttrs_HWML()) {
        auto argAttrRange = argAttrs.template getAsRange<DictionaryAttr>();
        result.append(argAttrRange.begin(), argAttrRange.end());
      } else {
        result.append($_op.getNumResults_HWML(),
                      DictionaryAttr::get(this->getOperation()->getContext()));
      }
    }

    /// Set the attributes held by the result at 'index'.
    void setResultAttrs_HWML(unsigned index, ArrayRef<NamedAttribute> attributes) {
      ::circt::hw::HWModuleLike_impl::setResultAttrs($_op, index, attributes);
    }

    /// Set the attributes held by the result at 'index'. `attributes` may be
    /// null, in which case any existing argument attributes are removed.
    void setResultAttrs_HWML(unsigned index, DictionaryAttr attributes) {
      ::circt::hw::HWModuleLike_impl::setResultAttrs($_op, index, attributes);
    }

    /// If the an attribute exists with the specified name, change it to the new
    /// value. Otherwise, add a new attribute with the specified name/value.
    void setResultAttr_HWML(unsigned index, StringAttr name, Attribute value) {
      ::circt::hw::HWModuleLike_impl::setResultAttr($_op, index, name, value);
    }
    void setResultAttr_HWML(unsigned index, StringRef name, Attribute value) {
      setResultAttr_HWML(index,
                    StringAttr::get(this->getOperation()->getContext(), name),
                    value);
    }

    /// Returns the dictionary attribute corresponding to the result at 'index'.
    /// If there are no result attributes at 'index', a null attribute is
    /// returned.
    DictionaryAttr getResultAttrDict_HWML(unsigned index) {
      assert(index < $_op.getNumResults_HWML() && "invalid result number");
      return ::circt::hw::HWModuleLike_impl::getResultAttrDict($_op, index);
    }

    /// Remove the attribute 'name' from the result at 'index'. Return the
    /// attribute that was erased, or nullptr if there was no attribute with
    /// such name.
    Attribute removeResultAttr(unsigned index, StringAttr name) {
      return ::circt::hw::HWModuleLike_impl::removeResultAttr($_op, index, name);
    }


  }];

  let verify = [{
    static_assert(
        ConcreteOp::template hasTrait<::mlir::SymbolOpInterface::Trait>(),
        "expected operation to be a symbol");
    return success();
  }];
}

def HWMutableModuleLike : OpInterface<"HWMutableModuleLike", [HWModuleLike]> {
  let cppNamespace = "circt::hw";
  let description = "Provide methods to mutate a module.";

  let methods = [

    InterfaceMethod<
    "Decode information about the input and output ports on this module.",
    "::circt::hw::ModulePortInfo", "getPorts", (ins),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{
      return getModulePortInfo($_op);
    }]>,

    InterfaceMethod<"Get a handle to a utility class which provides by-name lookup of port indices. The returned object does _not_ update if the module is mutated.",
    "::circt::hw::ModulePortLookupInfo", "getPortLookupInfo", (ins),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{
      return hw::ModulePortLookupInfo(
        $_op->getContext(),
        this->getPorts());
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
    "::llvm::StringRef", "getInstanceName", (ins)>,

    InterfaceMethod<"Get the name of the instance",
    "::mlir::StringAttr", "getInstanceNameAttr", (ins)>,

    InterfaceMethod<"Get the name of the instantiated module",
    "::llvm::StringRef", "getReferencedModuleName", (ins),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{ return $_op.getModuleName(); }]>,

    InterfaceMethod<"Get the name of the instantiated module",
    "::mlir::StringAttr", "getReferencedModuleNameAttr", (ins),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{ return $_op.getModuleNameAttr().getAttr(); }]>,

    InterfaceMethod<"Get the referenced module",
    "::mlir::Operation *", "getReferencedModule", (ins)>,
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
    InterfaceMethod<"Returns the name of the top-level inner symbol defined by this operation, if present.",
      "::mlir::StringAttr", "getInnerNameAttr", (ins), [{}],
      /*defaultImplementation=*/[{
        if (auto attr =
              this->getOperation()->template getAttrOfType<hw::InnerSymAttr>(
              circt::hw::InnerSymbolTable::getInnerSymbolAttrName()))
          return attr.getSymName();
        return {};
      }]
    >,
    InterfaceMethod<"Returns the name of the top-level inner symbol defined by this operation, if present.",
      "::std::optional<::mlir::StringRef>", "getInnerName", (ins), [{}],
      /*defaultImplementation=*/[{
        auto attr = this->getInnerNameAttr();
        return attr ? ::std::optional<StringRef>(attr.getValue()) : ::std::nullopt;
      }]
    >,
    InterfaceMethod<"Sets the name of the top-level inner symbol defined by this operation to the specified string, dropping any symbols on fields.",
      "void", "setInnerSymbol", (ins "::mlir::StringAttr":$name), [{}],
      /*defaultImplementation=*/[{
        this->getOperation()->setAttr(
            InnerSymbolTable::getInnerSymbolAttrName(), hw::InnerSymAttr::get(name));
      }]
    >,
    InterfaceMethod<"Sets the inner symbols defined by this operation.",
      "void", "setInnerSymbolAttr", (ins "::circt::hw::InnerSymAttr":$sym), [{}],
      /*defaultImplementation=*/[{
        if (sym && !sym.empty())
          this->getOperation()->setAttr(
              InnerSymbolTable::getInnerSymbolAttrName(), sym);
        else
          this->getOperation()->removeAttr(InnerSymbolTable::getInnerSymbolAttrName());
      }]
    >,
    InterfaceMethod<"Returns an InnerRef to this operation's top-level inner symbol, which must be present.",
      "::circt::hw::InnerRefAttr", "getInnerRef", (ins), [{}],
      /*defaultImplementation=*/[{
        auto *op = this->getOperation();
        return hw::InnerRefAttr::get(
            SymbolTable::getSymbolName(
                op->template getParentWithTrait<OpTrait::InnerSymbolTable>()),
            InnerSymbolTable::getInnerSymbol(op));
      }]
    >,
    InterfaceMethod<"Returns the InnerSymAttr representing all inner symbols defined by this operation.",
      "::circt::hw::InnerSymAttr", "getInnerSymAttr", (ins), [{}],
      /*defaultImplementation=*/[{
        return this->getOperation()->template getAttrOfType<hw::InnerSymAttr>(
              circt::hw::InnerSymbolTable::getInnerSymbolAttrName());
      }]
    >,
    // Ask an operation if per-field symbols are allowed.
    // Defaults to indicating they're allowed iff there's a defined target result,
    // but let operations answer this differently if for some reason that makes sense.
    StaticInterfaceMethod<"Returns whether per-field symbols are supported for this operation type.",
      "bool", "supportsPerFieldSymbols", (ins), [{}], /*defaultImplementation=*/[{
       return ConcreteOp::getTargetResultIndex().has_value();
    }]>,
    StaticInterfaceMethod<"Returns the index of the result the innner symbol targets, if applicable.  Per-field symbols are resolved into this.",
      "std::optional<size_t>", "getTargetResultIndex">,
    InterfaceMethod<"Returns the result the innner symbol targets, if applicable.  Per-field symbols are resolved into this.",
      "OpResult", "getTargetResult", (ins), [{}], /*defaultImplementation=*/[{
        auto idx = ConcreteOp::getTargetResultIndex();
        if (!idx)
          return {};
        return $_op->getResult(*idx);
      }]>,
  ];

  let verify = [{
    return verifyInnerSymAttr(cast<circt::hw::InnerSymbolOpInterface>(op));
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
