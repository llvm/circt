//===- PortsInterfaces.td - Ports+Symbols Interfaces -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the HWPorts and HWPortsSymbols interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_PORTSINTERFACES
#define CIRCT_DIALECT_HW_PORTSINTERFACES

include "mlir/IR/OpBase.td"

def HWPorts : OpInterface<"HWPorts"> {
  let cppNamespace = "circt::hw";
  let description = "Provide accessors for ports";

  let methods = [
    // Getters
    InterfaceMethod<"Get the number of ports",
    "size_t", "getNumPorts">,
  ];

}

def HWPortsSymbols : OpInterface<"HWPortsSymbols", [DeclareOpInterfaceMethods<HWPorts>]> {
  let cppNamespace = "circt::hw";
  let description = "Provide accessors for ports and their symbols";


  let methods = [
    // Getters
    InterfaceMethod<"Get a port symbol attribute",
    "::circt::hw::InnerSymAttr", "getPortSymbolAttr", (ins "size_t":$portIndex)>,

    InterfaceMethod<"Check if port has symbol attribute",
    "bool", "hasPortSymbolAttr", (ins "size_t":$portIndex), [{}], [{
      assert(portIndex < $_op.getNumPorts());
      auto symAttr = $_op.getPortSymbolAttr(portIndex);
      return symAttr && !symAttr.empty();
    }]>,

    // Setters
    InterfaceMethod<"Set the symbols for a port including its fields", "void",
    "setPortSymbolAttr", (ins "size_t":$portIndex, "circt::hw::InnerSymAttr":$symbol)>,

    // Convenience methods wrapping setPortSymbolAttr.
    InterfaceMethod<"Set a port's top-level symbol to the specified string, dropping any symbols on its fields", "void",
    "setPortSymbolString", (ins "size_t":$portIndex, "StringRef":$symbol), [{}], [{
      $_op.setPortSymbolStringAttr(portIndex, StringAttr::get($_op.getContext(),
        symbol));
    }]>,

    InterfaceMethod<"Set a port's top-level symbol to the specified string, dropping any symbols on its fields", "void",
    "setPortSymbolStringAttr", (ins "size_t":$portIndex, "StringAttr":$symbol), [{}], [{
      return $_op.setPortSymbolAttr(portIndex, hw::InnerSymAttr::get(symbol));
    }]>,
  ];

}

#endif // CIRCT_DIALECT_HW_PORTSINTERFACES
