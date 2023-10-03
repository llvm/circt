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
include "circt/Support/InstanceGraphInterface.td"

def PortList : OpInterface<"PortList", []> {
  let cppNamespace = "circt::hw";
  let description = "Operations which produce a unified port list representation";
  let methods = [
    InterfaceMethod<"Get port list",
    "SmallVector<::circt::hw::PortInfo>", "getPortList", (ins)>,

    InterfaceMethod<"Get port list",
    "::circt::hw::PortInfo", "getPort", (ins "size_t":$idx)>,

    InterfaceMethod<"Get the port a specific input",
    "size_t", "getPortIdForInputId", (ins "size_t":$idx)>,

    InterfaceMethod<"Get the port a specific output",
    "size_t", "getPortIdForOutputId", (ins "size_t":$idx)>,

    InterfaceMethod<"Get the number of ports",
    "size_t", "getNumPorts", (ins)>,
    InterfaceMethod<"Get the number of input ports",
    "size_t", "getNumInputPorts", (ins)>,
    InterfaceMethod<"Get the number of output ports",
    "size_t", "getNumOutputPorts", (ins)>,
  ];
}

def HWModuleLike : OpInterface<"HWModuleLike", [
  Symbol, PortList, InstanceGraphModuleOpInterface]> {
  let cppNamespace = "circt::hw";
  let description = "Provide common module information.";

  let methods = [
    InterfaceMethod<"Get the module type",
    "::circt::hw::ModuleType", "getHWModuleType", (ins)>,

    InterfaceMethod<"Get the port Attributes",
    "SmallVector<Attribute>", "getAllPortAttrs", (ins)>,

    InterfaceMethod<"Set the port Attributes",
    "void", "setAllPortAttrs", (ins "ArrayRef<Attribute>":$attrs)>,

    InterfaceMethod<"Remove the port Attributes",
    "void", "removeAllPortAttrs", (ins)>,

    InterfaceMethod<"Get the port Locations",
    "SmallVector<Location>", "getAllPortLocs", (ins)>,

    InterfaceMethod<"Set the port Locations",
    "void", "setAllPortLocs", (ins "ArrayRef<Location>":$locs)>,

    InterfaceMethod<"Set the module type (and port names)",
    "void", "setHWModuleType", (ins "::circt::hw::ModuleType":$type)>,

    InterfaceMethod<"Set the port names",
    "void", "setAllPortNames", (ins "ArrayRef<Attribute>":$names)>,

  ];

  let extraSharedClassDeclaration = [{
    /// Attribute name for port symbols.
    static StringRef getPortSymbolAttrName() {
      return "hw.exportPort";
    }

    /// Return the region containing the body of this function.
    Region &getModuleBody() { return $_op->getRegion(0); }
    Block *getBodyBlock() {
      Region &body = getModuleBody();
      if (body.empty())
        return nullptr;
      return &body.front();
    }

    /// Returns the entry block argument at the given index.
    BlockArgument getArgumentForInput(unsigned idx) {
      return $_op.getModuleBody().getArgument(idx);
    }

    /// Returns the entry block argument for the given port.  May be null.
    BlockArgument getArgumentForPort(unsigned idx) {
      return $_op.getModuleBody().getArgument($_op.getHWModuleType().getInputIdForPortId(idx));
    }

    SmallVector<mlir::Type> getPortTypes() {
      return $_op.getHWModuleType().getPortTypes();
    }

    SmallVector<mlir::Type> getInputTypes() {
      return $_op.getHWModuleType().getInputTypes();
    }

    SmallVector<mlir::Type> getOutputTypes() {
      return $_op.getHWModuleType().getOutputTypes();
    }

    /// Return the set of names on input and inout ports
    SmallVector<StringAttr> getInputNamesStr() {
      return $_op.getHWModuleType().getInputNamesStr();
    }

    /// Return the set of names on output ports
    SmallVector<StringAttr> getOutputNamesStr() {
      return $_op.getHWModuleType().getOutputNamesStr();
    }

    /// Return the set of names on input and inout ports
    SmallVector<Attribute> getInputNames() {
      return $_op.getHWModuleType().getInputNames();
    }

    /// Return the set of names on output ports
    SmallVector<Attribute> getOutputNames() {
      return $_op.getHWModuleType().getOutputNames();
    }

    void setInputNames(ArrayRef<Attribute> names) {
      SmallVector<Attribute> newNames(names.begin(), names.end());
      auto resNames = $_op.getOutputNames();
      newNames.append(resNames.begin(), resNames.end());
      $_op.setAllPortNames(newNames);
    }

    void setOutputNames(ArrayRef<Attribute> names) {
      SmallVector<Attribute> newNames = $_op.getInputNames();
      newNames.append(names.begin(), names.end());
      $_op.setAllPortNames(newNames);
    }

    // Get the name for the specified input or inout port
    StringRef getPortName(size_t idx) {
      return $_op.getHWModuleType().getPortName(idx);
    }

    // Get the name for the specified input or inout port
    StringRef getInputName(size_t idx) {
      return $_op.getHWModuleType().getInputName(idx);
    }

    // Get the name for the specified output port
    StringRef getOutputName(size_t idx) {
      return $_op.getHWModuleType().getOutputName(idx);
    }

    StringAttr getInputNameAttr(size_t idx) {
      return $_op.getHWModuleType().getInputNameAttr(idx);
    }

    StringAttr getOutputNameAttr(size_t idx) {
      return $_op.getHWModuleType().getOutputNameAttr(idx);
    }

    Attribute getPortAttrs(size_t idx) {
      return $_op.getAllPortAttrs()[idx];
    }

    SmallVector<Attribute> getAllInputAttrs() {
      auto attrs = $_op.getAllPortAttrs();
      auto num = $_op.getNumInputPorts();
      SmallVector<Attribute> retval(num);
      auto modType = $_op.getHWModuleType();
      for (unsigned x = 0; x < num; ++x)
        retval[x] = attrs[modType.getPortIdForInputId(x)];
      return retval;
    }

    SmallVector<Attribute> getAllOutputAttrs() {
      auto attrs = $_op.getAllPortAttrs();
      auto num = $_op.getNumOutputPorts();
      SmallVector<Attribute> retval(num);
      auto modType = $_op.getHWModuleType();
      for (unsigned x = 0; x < num; ++x)
        retval[x] = attrs[modType.getPortIdForOutputId(x)];
      return retval;
    }

    void setAllInputAttrs(ArrayRef<Attribute> attrs) {
      SmallVector<Attribute> retval(attrs.begin(), attrs.end());
      auto resAttrs = $_op.getAllOutputAttrs();
      retval.append(resAttrs.begin(), resAttrs.end());
      $_op.setAllPortAttrs(retval);
    }

    void setAllOutputAttrs(ArrayRef<Attribute> attrs) {
      SmallVector<Attribute> retval = $_op.getAllInputAttrs();
      retval.append(attrs.begin(), attrs.end());
      $_op.setAllPortAttrs(retval);
    }

    Attribute getInputAttrs(size_t idx) {
      return $_op.getAllPortAttrs()[$_op.getPortIdForInputId(idx)];
    }

    Attribute getOutputAttrs(size_t idx) {
      return $_op.getAllPortAttrs()[$_op.getPortIdForOutputId(idx)];
    }

    void setPortAttrs(size_t idx, DictionaryAttr attr) {
      auto attrs = $_op.getAllPortAttrs();
      attrs[idx] = attr;
      $_op.setAllPortAttrs(attrs);
    }

    void setPortAttr(size_t idx, StringAttr name, Attribute value) {
      auto attrs = $_op.getAllPortAttrs();
      NamedAttrList pattr(cast<DictionaryAttr>(attrs[idx]));
      Attribute oldValue;
      if (!value)
        oldValue = pattr.erase(name);
      else
        oldValue = pattr.set(name, value);
      if (oldValue != value) {
        attrs[idx] = pattr.getDictionary($_op.getContext());
        $_op.setAllPortAttrs(attrs);
      }
    }

    void setPortAttrs(StringAttr attrName, ArrayRef<Attribute> newAttrs) {
      auto attrs = $_op.getAllPortAttrs();
      auto ctxt = $_op.getContext();
      assert(newAttrs.size() == attrs.size());
      for (size_t idx = 0, e = attrs.size(); idx != e; ++idx) {
        NamedAttrList pattr(cast_or_null<mlir::DictionaryAttr>(attrs[idx]));
        auto newAttr = newAttrs[idx];
        if (newAttr)
          pattr.set(attrName, newAttr);
        else
          pattr.erase(attrName);
        attrs[idx] = pattr.getDictionary(ctxt);
      }
      $_op.setAllPortAttrs(attrs);
    }

    Location getPortLoc(size_t idx) {
      return $_op.getAllPortLocs()[idx];
    }

    void setPortLoc(size_t idx, Location loc) {
      auto locs = $_op.getAllPortLocs();
      locs[idx] = loc;
      return $_op.setAllPortLocs(locs);
    }

    Location getInputLoc(size_t idx) {
      return $_op.getAllPortLocs()[$_op.getPortIdForInputId(idx)];
    }

    Location getOutputLoc(size_t idx) {
      return $_op.getAllPortLocs()[$_op.getPortIdForOutputId(idx)];
    }

    SmallVector<Location> getInputLocs() {
      auto locs = $_op.getAllPortLocs();
      SmallVector<Location> retval;
      for (unsigned x = 0, e = $_op.getNumInputPorts(); x < e; ++x)
        retval.push_back(locs[$_op.getPortIdForInputId(x)]);
      return retval;
    }

    ArrayAttr getInputLocsAttr() {
      auto locs = $_op.getAllPortLocs();
      SmallVector<Attribute> retval;
      for (unsigned x = 0, e = $_op.getNumInputPorts(); x < e; ++x)
        retval.push_back(locs[$_op.getPortIdForInputId(x)]);
      return ArrayAttr::get($_op->getContext(), retval);
    }

    void setInputLocs(ArrayRef<Location> inAttrs) {
      assert(inAttrs.size() == $_op.getNumInputPorts());
      auto outAttrs = getOutputLocs();
      SmallVector<Location> attrs;
      attrs.append(inAttrs.begin(), inAttrs.end());
      attrs.append(outAttrs.begin(), outAttrs.end());
      $_op.setAllPortLocs(attrs);
    }

    SmallVector<Location> getOutputLocs() {
      auto locs = $_op.getAllPortLocs();
      SmallVector<Location> retval;
      for (unsigned x = 0, e = $_op.getNumOutputPorts(); x < e; ++x)
        retval.push_back(locs[$_op.getPortIdForOutputId(x)]);
      return retval;
    }

    ArrayAttr getOutputLocsAttr() {
      auto locs = $_op.getAllPortLocs();
      SmallVector<Attribute> retval;
      for (unsigned x = 0, e = $_op.getNumOutputPorts(); x < e; ++x)
        retval.push_back(locs[$_op.getPortIdForOutputId(x)]);
      return ArrayAttr::get($_op->getContext(), retval);
    }

    void setOutputLocs(ArrayRef<Location> outAttrs) {
      assert(outAttrs.size() == $_op.getNumOutputPorts());
      auto inAttrs = getInputLocs();
      SmallVector<Location> attrs;
      attrs.append(inAttrs.begin(), inAttrs.end());
      attrs.append(outAttrs.begin(), outAttrs.end());
      $_op.setAllPortLocs(attrs);
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

    InterfaceMethod<"Get a handle to a utility class which provides by-name lookup of port indices. The returned object does _not_ update if the module is mutated.",
    "::circt::hw::ModulePortLookupInfo", "getPortLookupInfo", (ins),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{
      return hw::ModulePortLookupInfo(
        $_op->getContext(),
        $_op.getPortList());
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


def HWInstanceLike : OpInterface<"HWInstanceLike", [
    PortList, InstanceGraphInstanceOpInterface]> {
  let cppNamespace = "circt::hw";
  let description = "Provide common  module information.";

  let methods = [
    InterfaceMethod<"Get the referenced module via a HW symbol cache.",
    "::mlir::Operation *", "getReferencedModuleCached", (ins
      "const ::circt::hw::HWSymbolCache *":$cache),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{
      return hw::instance_like_impl::getReferencedModule(
          cache, $_op, $_op.getModuleNameAttr());
    }]>,
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
