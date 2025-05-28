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
include "circt/Dialect/Emit/EmitOpInterfaces.td"
include "circt/Support/InstanceGraphInterface.td"

def PortList : OpInterface<"PortList", []> {
  let cppNamespace = "circt::hw";
  let description = "Operations which produce a unified port list representation";
  let methods = [
    /// This is slow.  Do not use it repeatedly.
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

    InterfaceMethod<"Get the port Attributes.  This will return either an empty array or an array of size numPorts.",
    "ArrayRef<Attribute>", "getAllPortAttrs", (ins)>,

    InterfaceMethod<"Set the port Attributes",
    "void", "setAllPortAttrs", (ins "ArrayRef<Attribute>":$attrs)>,

    InterfaceMethod<"Remove the port Attributes",
    "void", "removeAllPortAttrs", (ins)>,

    InterfaceMethod<"Get the port Locations",
    "SmallVector<Location>", "getAllPortLocs", (ins)>,

    InterfaceMethod<"Set the port Locations",
    "void", "setAllPortLocsAttrs", (ins "ArrayRef<Attribute>":$locs)>,

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
    SmallVector<Attribute> getInputNames() {
      return $_op.getHWModuleType().getInputNames();
    }

    /// Return the set of names on output ports
    SmallVector<Attribute> getOutputNames() {
      return $_op.getHWModuleType().getOutputNames();
    }

    void setInputNames(ArrayRef<Attribute> names) {
      auto type = $_op.getHWModuleType();
      SmallVector<Attribute> newNames(type.getPortNames());
      for (size_t idx = 0, e = names.size(); idx != e; ++idx)
        newNames[type.getPortIdForInputId(idx)] = names[idx];
      $_op.setAllPortNames(newNames);
    }

    void setOutputNames(ArrayRef<Attribute> names) {
      auto type = $_op.getHWModuleType();
      SmallVector<Attribute> newNames(type.getPortNames());
      for (size_t idx = 0, e = names.size(); idx != e; ++idx)
        newNames[type.getPortIdForOutputId(idx)] = names[idx];
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
      auto attrs = $_op.getAllPortAttrs();
      if (attrs.empty())
        return nullptr;
      return attrs[idx];
    }

    SmallVector<Attribute> getAllInputAttrs() {
      auto attrs = $_op.getAllPortAttrs();
      if (attrs.empty())
        return {};
      auto num = $_op.getNumInputPorts();
      SmallVector<Attribute> retval(num);
      auto modType = $_op.getHWModuleType();
      for (unsigned x = 0; x < num; ++x)
        retval[x] = attrs[modType.getPortIdForInputId(x)];
      return retval;
    }

    SmallVector<Attribute> getAllOutputAttrs() {
      auto attrs = $_op.getAllPortAttrs();
      if (attrs.empty())
        return {};
      auto num = $_op.getNumOutputPorts();
      SmallVector<Attribute> retval(num);
      auto modType = $_op.getHWModuleType();
      for (unsigned x = 0; x < num; ++x)
        retval[x] = attrs[modType.getPortIdForOutputId(x)];
      return retval;
    }

    void setAllInputAttrs(ArrayRef<Attribute> attrs) {
      auto type = $_op.getHWModuleType();
      SmallVector<Attribute> newAttrs($_op.getAllPortAttrs());
      if (newAttrs.empty())
        newAttrs.resize($_op.getNumPorts());
      for (size_t idx = 0, e = attrs.size(); idx != e; ++idx)
        newAttrs[type.getPortIdForInputId(idx)] = attrs[idx];
      $_op.setAllPortAttrs(newAttrs);
    }

    void setAllOutputAttrs(ArrayRef<Attribute> attrs) {
      auto type = $_op.getHWModuleType();
      SmallVector<Attribute> newAttrs($_op.getAllPortAttrs());
      if (newAttrs.empty())
        newAttrs.resize($_op.getNumPorts());
      for (size_t idx = 0, e = attrs.size(); idx != e; ++idx)
        newAttrs[type.getPortIdForOutputId(idx)] = attrs[idx];
      $_op.setAllPortAttrs(newAttrs);
    }

    Attribute getInputAttrs(size_t idx) {
      auto allPortAttrs = $_op.getAllPortAttrs();
      assert(idx < $_op.getNumInputPorts()
        && "Out of range input port index");
      if (allPortAttrs.empty())
        return DictionaryAttr::get($_op.getContext());
      return allPortAttrs[$_op.getPortIdForInputId(idx)];
    }

    Attribute getOutputAttrs(size_t idx) {
      auto allPortAttrs = $_op.getAllPortAttrs();
      assert(idx < $_op.getNumOutputPorts()
        && "Out of range output port index");
      if (allPortAttrs.empty())
        return DictionaryAttr::get($_op.getContext());
      return allPortAttrs[$_op.getPortIdForOutputId(idx)];
    }

    void setPortAttrs(size_t idx, DictionaryAttr attr) {
      auto ctxt = $_op.getContext();
      auto allPortAttrsArray = $_op.getAllPortAttrs();
      SmallVector<Attribute> attrs;
      if (!allPortAttrsArray.empty()) // Copy the existing array for modification
        attrs = SmallVector<Attribute>(allPortAttrsArray);
      else if (!attr.empty()) // Create a new array filled with empty dictionaries
        attrs = SmallVector<Attribute>($_op.getNumPorts(), DictionaryAttr::get(ctxt));
      else // Existing empty array stays empty
        return;
      attrs[idx] = attr;
      $_op.setAllPortAttrs(attrs);
    }

    void setPortAttr(size_t idx, StringAttr name, Attribute value) {
      auto ctxt = $_op.getContext();
      auto allPortAttrsArray = $_op.getAllPortAttrs();
      SmallVector<Attribute> attrs;
      if (!allPortAttrsArray.empty()) // Copy the existing array for modification
        attrs = SmallVector<Attribute>(allPortAttrsArray);
      else if (value)  // Create a new array filled with empty dictionaries
        attrs = SmallVector<Attribute>($_op.getNumPorts(), DictionaryAttr::get(ctxt));
      else // Existing empty array stays empty
        return;
      NamedAttrList pattr(cast<DictionaryAttr>(attrs[idx]));
      Attribute oldValue;
      if (!value)
        oldValue = pattr.erase(name);
      else
        oldValue = pattr.set(name, value);
      if (oldValue != value) {
        attrs[idx] = pattr.getDictionary(ctxt);
        $_op.setAllPortAttrs(attrs);
      }
    }

    void setPortAttrs(StringAttr attrName, ArrayRef<Attribute> newAttrs) {
      auto ctxt = $_op.getContext();
      auto allPortAttrsArray = $_op.getAllPortAttrs();
      SmallVector<Attribute> attrs;
      if (!allPortAttrsArray.empty()) // Copy the existing array for modification
        attrs = SmallVector<Attribute>(allPortAttrsArray);
      else if (!newAttrs.empty()) // Create a new array filled with empty dictionaries
        attrs = SmallVector<Attribute>($_op.getNumPorts(), DictionaryAttr::get(ctxt));
      else // Existing empty array stays empty
        return;
      assert(newAttrs.size() == attrs.size());
      for (size_t idx = 0, e = attrs.size(); idx != e; ++idx) {
        NamedAttrList pattr(cast<mlir::DictionaryAttr>(attrs[idx]));
        auto newAttr = newAttrs[idx];
        if (newAttr)
          pattr.set(attrName, newAttr);
        else
          pattr.erase(attrName);
        attrs[idx] = pattr.getDictionary(ctxt);
      }
      $_op.setAllPortAttrs(attrs);
    }

    void setAllPortLocs(ArrayRef<Location> locs) {
      std::vector<Attribute> nLocs(locs.begin(), locs.end());
      $_op.setAllPortLocsAttrs(nLocs);
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

    void setInputLocs(ArrayRef<Location> inLocs) {
      assert(inLocs.size() == $_op.getNumInputPorts());
      auto type = $_op.getHWModuleType();
      SmallVector<Location> newLocs($_op.getAllPortLocs());
      for (size_t idx = 0, e = inLocs.size(); idx != e; ++idx)
        newLocs[type.getPortIdForInputId(idx)] = inLocs[idx];
      $_op.setAllPortLocs(newLocs);

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

    void setOutputLocs(ArrayRef<Location> outLocs) {
      assert(outLocs.size() == $_op.getNumOutputPorts());
      auto type = $_op.getHWModuleType();
      SmallVector<Location> newLocs($_op.getAllPortLocs());
      for (size_t idx = 0, e = outLocs.size(); idx != e; ++idx)
        newLocs[type.getPortIdForOutputId(idx)] = outLocs[idx];
      $_op.setAllPortLocs(newLocs);
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

def HWEmittableModuleLike : OpInterface<"HWEmittableModuleLike", [HWModuleLike,
                                        Emittable]> {
  let cppNamespace = "::circt::hw";
  let description = [{
    This interface indicates that the module like op is emittable in SV and
    requires SV legalization on its body.
  }];
}


def HWInstanceLike : OpInterface<"HWInstanceLike", [
    InstanceGraphInstanceOpInterface]> {
  let cppNamespace = "circt::hw";
  let description = "Provide common instance information.";

  // Note that the single-element getters and setters could be implemented as
  // shared declarations using the interface methods that operate on the entire
  // array or vice versa.  However, depending on how the names are stored in the
  // instance operation, they can be implemented more efficiently by giving the
  // operation the freedom to define all those methods (e.g., by avoiding
  // construction of unnecessary intermedate array attributes).
  let methods = [
    InterfaceMethod<[{
      Return the name of the specified input port or null if it cannot be
      determined.
    }], "::mlir::StringAttr", "getInputName", (ins "size_t":$idx),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{
      return ::circt::hw::instance_like_impl::getName(
        $_op.getArgNames(), idx);
    }]>,

    InterfaceMethod<[{
      Return the name of the specified result or null if it cannot be
      determined.
    }], "::mlir::StringAttr", "getOutputName", (ins "size_t":$idx),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{
      return ::circt::hw::instance_like_impl::getName(
        $_op.getResultNames(), idx);
    }]>,

    InterfaceMethod<[{
      Change the name of the specified input port.
    }], "void", "setInputName",
    (ins "size_t":$idx, "::mlir::StringAttr":$name),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{
      $_op.setArgNamesAttr(::circt::hw::instance_like_impl::updateName(
        $_op.getArgNames(), idx, name));
    }]>,

    InterfaceMethod<[{
      Change the name of the specified output port.
    }], "void", "setOutputName",
    (ins "size_t":$idx, "::mlir::StringAttr":$name),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{
      $_op.setResultNamesAttr(::circt::hw::instance_like_impl::updateName(
        $_op.getResultNames(), idx, name));
    }]>,

    InterfaceMethod<[{
      Return the names of all input ports. If the instance operation stores the
      names in an ArrayAttr this can avoid attribute constructions.
    }],
    "::mlir::ArrayAttr", "getInputNames", (ins),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{
      return $_op.getArgNamesAttr();
    }]>,

    InterfaceMethod<[{
      Return the name of all ouput ports. If the instance operation stores the
      names in an ArrayAttr this can avoid attribute constructions.
    }],
    "::mlir::ArrayAttr", "getOutputNames", (ins),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{
      return $_op.getResultNamesAttr();
    }]>,

    InterfaceMethod<[{
      Change the names of all input ports. If all names have to be changed, this
      can avoid repeated intermediate attribute constructions.
    }], "void", "setInputNames", (ins "::mlir::ArrayAttr":$names),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{
      $_op.setArgNamesAttr(names);
    }]>,

    InterfaceMethod<[{
      Change the names of all output ports. If all names have to be changed, this
      can avoid repeated intermediate attribute constructions.
    }], "void", "setOutputNames", (ins "::mlir::ArrayAttr":$names),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{
      $_op.setResultNamesAttr(names);
    }]>,

    InterfaceMethod<"True if this instance is a phony placeholder",
      "bool", "getDoNotPrint", (ins)
    >
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

def CombDataflow : OpInterface<"CombDataFlow"> {
  let cppNamespace = "circt::hw";
  let description = [{
    This interface is used for specifying the combinational dataflow that exists
    in the results and operands of an operation.
    Any operation that doesn't implement this interface is assumed to have a
    combinational dependence from each operand to each result.
  }];
  let methods = [
    InterfaceMethod<[{
     Get the combinational dataflow relations between the operands and the results.
     This returns a pair of ground type fieldrefs. The first element is the destination
     and the second is the source of the dependence. The default implementation returns
     an empty list, which implies that the operation is not combinational.}],
     "llvm::SmallVector<std::pair<circt::FieldRef, circt::FieldRef>>", "computeDataFlow",
     (ins)>,
  ];
}

#endif
