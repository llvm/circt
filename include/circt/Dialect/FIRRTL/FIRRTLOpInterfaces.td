//===- FIRRTLOpInterfaces.td - Operation Interfaces --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the FIRRTL operation interfaces.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"

def FModuleLike : OpInterface<"FModuleLike"> {
  let cppNamespace = "circt::firrtl";
  let description = "Provide common  module information.";
  let methods = [

    //===------------------------------------------------------------------===//
    // Module Name
    //===------------------------------------------------------------------===//

    InterfaceMethod<"Get the module name",
    "StringRef", "moduleName", (ins),
    /*methodBody=*/[{ return $_op.getName(); }]>,

    //===------------------------------------------------------------------===//
    // Port Directions
    //===------------------------------------------------------------------===//

    InterfaceMethod<"Get the port directions attribute",
    "IntegerAttr", "getPortDirectionsAttr", (ins), [{}],
    /*defaultImplementation=*/[{
      return $_op->template
        getAttrOfType<IntegerAttr>(FModuleLike::getPortDirectionsAttrName());
    }]>,

    InterfaceMethod<"Get the port directions",
    "APInt", "getPortDirections", (ins), [{}],
    /*defaultImplementation=*/[{
      return $_op.getPortDirectionsAttr().getValue();
    }]>,

    InterfaceMethod<"Get a port direction",
    "Direction", "getPortDirection", (ins "size_t":$portIndex), [{}],
    /*defaultImplementation=*/[{
      return direction::get($_op.getPortDirections()[portIndex]);
    }]>,

    //===------------------------------------------------------------------===//
    // Port Names
    //===------------------------------------------------------------------===//

    InterfaceMethod<"Get the port names attribute",
    "ArrayAttr", "getPortNamesAttr", (ins), [{}],
    /*defaultImplementation=*/[{
      return $_op->template
        getAttrOfType<ArrayAttr>(FModuleLike::getPortNamesAttrName());
    }]>,

    InterfaceMethod<"Get the port names",
    "ArrayRef<Attribute>", "getPortNames", (ins), [{}],
    /*defaultImplementation=*/[{
      return $_op.getPortNamesAttr().getValue();
    }]>,

    InterfaceMethod<"Get a port name",
    "StringAttr", "getPortNameAttr", (ins "size_t":$portIndex), [{}],
    /*defaultImplementation=*/[{
      return $_op.getPortNames()[portIndex].template cast<StringAttr>();
    }]>,

    InterfaceMethod<"Get a port name",
    "StringRef", "getPortName", (ins "size_t":$portIndex), [{}],
    /*defaultImplementation=*/[{
      return $_op.getPortNameAttr(portIndex).getValue();
    }]>,

    //===------------------------------------------------------------------===//
    // Port Types
    //===------------------------------------------------------------------===//

    InterfaceMethod<"Get the port types attribute",
    "ArrayAttr", "getPortTypesAttr", (ins), [{}],
    /*defaultImplementation=*/[{
      return $_op->template
        getAttrOfType<ArrayAttr>(FModuleLike::getPortTypesAttrName());
    }]>,

    InterfaceMethod<"Get the port types",
    "ArrayRef<Attribute>", "getPortTypes", (ins), [{}],
    /*defaultImplementation=*/[{
      return $_op.getPortTypesAttr().getValue();
    }]>,

    InterfaceMethod<"Get a port type",
    "TypeAttr", "getPortTypeAttr", (ins "size_t":$portIndex), [{}],
    /*defaultImplementation=*/[{
      return $_op.getPortTypes()[portIndex].template cast<TypeAttr>();
    }]>,

    InterfaceMethod<"Get a port type",
    "FIRRTLType", "getPortType", (ins "size_t":$portIndex), [{}],
    /*defaultImplementation=*/[{
      auto typeAttr = $_op.getPortTypeAttr(portIndex);
      return typeAttr.getValue().template cast<FIRRTLType>();
    }]>,

    //===------------------------------------------------------------------===//
    // Port Annotations
    //===------------------------------------------------------------------===//

    InterfaceMethod<"Get the port annotations attribute",
    "ArrayAttr", "getPortAnnotationsAttr", (ins), [{}],
    /*defaultImplementation=*/[{
      return $_op->template
        getAttrOfType<ArrayAttr>(FModuleLike::getPortAnnotationsAttrName());
    }]>,

    InterfaceMethod<"Get the port annotations attribute",
    "ArrayRef<Attribute>", "getPortAnnotations", (ins), [{}],
    /*defaultImplementation=*/[{
      return $_op.getPortAnnotationsAttr().getValue();
    }]>,

    InterfaceMethod<"Get a port's annotations attribute",
    "ArrayAttr", "getAnnotationsAttrForPort", (ins "size_t":$portIndex), [{}],
    /*defaultImplementation=*/[{
      return $_op.getPortAnnotations()[portIndex].template cast<ArrayAttr>();
    }]>,

    InterfaceMethod<"Get a port's annotations",
    "ArrayRef<Attribute>", "getAnnotationsForPort",
    (ins "size_t":$portIndex), [{}],
    /*defaultImplementation=*/[{
      return $_op.getAnnotationsAttrForPort(portIndex).getValue();
    }]>,

    //===------------------------------------------------------------------===//
    // Port Symbols
    //===------------------------------------------------------------------===//

    // Getters
    InterfaceMethod<"Get the port symbols attribute",
    "ArrayAttr", "getPortSymbolsAttr", (ins), [{}], [{
      return $_op->template
        getAttrOfType<ArrayAttr>(FModuleLike::getPortSymbolsAttrName());
    }]>,

    InterfaceMethod<"Get the port symbols",
    "ArrayRef<Attribute>", "getPortSymbols", (ins), [{}], [{
      return $_op.getPortSymbolsAttr().getValue();
    }]>,

    InterfaceMethod<"Get a port's symbol attribute",
    "StringAttr", "getPortSymbolAttr", (ins "size_t":$portIndex), [{}], [{
      return $_op.getPortSymbols()[portIndex].template cast<StringAttr>();
    }]>,

    InterfaceMethod<"Get a port's symbol",
    "StringRef", "getPortSymbol", (ins "size_t":$portIndex), [{}], [{
      return $_op.getPortSymbolAttr(portIndex).getValue();
    }]>,

    // Setters
    InterfaceMethod<"Set the port symbols attribute", "void",
    "setPortSymbolsAttr", (ins "ArrayAttr":$symbols), [{}], [{
      if (llvm::all_of(symbols.getValue(), [](Attribute symbol) {
         return symbol.cast<StringAttr>().getValue().empty(); }))
        symbols = ArrayAttr::get($_op.getContext(), {});
      assert(symbols.getValue().empty() ||
             symbols.getValue().size() == $_op.getNumPorts());
      $_op->setAttr(FModuleLike::getPortSymbolsAttrName(), symbols);
    }]>,

    InterfaceMethod<"Set the port symbols", "void",
    "setPortSymbols", (ins "ArrayRef<Attribute>":$symbols), [{}], [{
      if (llvm::all_of(symbols, [](Attribute symbol) {
         return symbol.cast<StringAttr>().getValue().empty(); }))
        symbols = {};
      assert(symbols.empty() || symbols.size() == $_op.getNumPorts());
      $_op.setPortSymbolsAttr(ArrayAttr::get($_op.getContext(), symbols));
    }]>,

    InterfaceMethod<"Set a port's symbol attribute", "void",
    "setPortSymbolAttr", (ins "size_t":$portIndex, "StringAttr":$symbol), [{}],
    [{
      SmallVector<Attribute> symbols($_op.getPortSymbols().begin(),
                                     $_op.getPortSymbols().end());
      if (symbols.empty()) {
        auto emptyString = StringAttr::get($_op.getContext(), "");
        symbols.resize($_op.getNumPorts(), emptyString);
      }
      assert(symbols.size() == $_op.getNumPorts());
      symbols[portIndex] = symbol;
      $_op.setPortSymbols(symbols);
    }]>,

    InterfaceMethod<"Set a port's symbol", "void",
    "setPortSymbol", (ins "size_t":$portIndex, "StringRef":$symbol), [{}], [{
      $_op.setPortSymbolAttr(portIndex, StringAttr::get($_op.getContext(),
        symbol));
    }]>,

    //===------------------------------------------------------------------===//
    // All Port Information
    //===------------------------------------------------------------------===//

    InterfaceMethod<"Get the number of ports",
    "size_t", "getNumPorts", (ins), [{}],
    /*defaultImplementation=*/[{
      return $_op.getPortTypesAttr().size();
    }]>,

    InterfaceMethod<"Get information about all ports",
    "SmallVector<PortInfo>", "getPorts">
  ];

  let extraClassDeclaration = [{
    /// Get the attribute name for port directions.
    static StringRef getPortDirectionsAttrName() {
      return "portDirections";
    }

    /// Get the attribute name for port names",
    static StringRef getPortNamesAttrName() {
      return "portNames";
    }

    /// Get the attribute name for port types.
    static StringRef getPortTypesAttrName() {
      return "portTypes";
    }

    /// Get the attribute name for port annotations.
    static StringRef getPortAnnotationsAttrName() {
      return "portAnnotations";
    }

    /// Get the attribute name for port symbols.
    static StringRef getPortSymbolsAttrName() {
      return "portSyms";
    }
  }];

  let verify = [{
    static_assert(
        ConcreteOp::template hasTrait<::mlir::SymbolOpInterface::Trait>(),
        "expected operation to be a symbol");
    return verifyModuleLikeOpInterface(op);
  }];
}

