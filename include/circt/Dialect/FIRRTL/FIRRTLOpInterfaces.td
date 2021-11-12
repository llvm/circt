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

def FlowKind : OpInterface<"FlowKind"> {
  let cppNamespace = "circt::firrtl";
  let description = [{
    Allows operations to specify what kind of flow they have.  When an operation
    does not implement this interface, they are assumed to have "source" flow.
  }];
  let methods = [
    InterfaceMethod<"Get the flow of a value",
    "::circt::firrtl::FlowResult", "getFlow",
    (ins "::mlir::OpResult":$result),
    /*methodBody=*/[{
      // Ask the op implementation what the flow is.
      return $_op.getFlow(result);
    }], /*defaultImplementation=*/[{
      // By default all results arguments are source flow.
      return ::circt::firrtl::Flow::Source;
    }]>,

    InterfaceMethod<"Get the flow of a block argument.",
    "::circt::firrtl::FlowResult", "getBlockArgFlow",
    (ins "::mlir::BlockArgument":$arg),
    /*methodBody=*/[{
      // Ask the op implementation what the flow is.
      return $_op.getBlockArgFlow(arg);
    }], /*defaultImplementation=*/[{
      // By default all block arguments are source flow.
      return ::circt::firrtl::Flow::Source;
    }]>
  ];
  let extraClassDeclaration = [{
    /// Compute the flow for a Value, \p val, as determined by the FIRRTL
    /// specification.  This recursively walks backwards from \p val to the
    /// declaration.  The resulting flow is a combination of the declaration
    /// flow (output ports and instance inputs are sinks, registers and wires
    /// are duplex, anything else is a source) and the number of intermediary
    /// flips. An even number of flips will result in the same flow as the
    /// declaration. An odd number of flips will result in reversed flow being
    /// returned.  The reverse of source is sink.  The reverse of sink is
    /// source.  The reverse of duplex is duplex.
    static Flow get(Value value);
  }];
}

/// Convenience helper for overriding the result flow kind.
def ResultFlowKind : OpTraitList<[
  DeclareOpInterfaceMethods<FlowKind, ["getFlow"]>]>;

/// Convenience helper for overriding the block argument flow kind.
def BlockArgFlowKind : OpTraitList<[
  DeclareOpInterfaceMethods<FlowKind, ["getBlockArgFlow"]>]>;

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
  }];

  let verify = [{
    static_assert(
        ConcreteOp::template hasTrait<::mlir::SymbolOpInterface::Trait>(),
        "expected operation to be a symbol");
    return verifyModuleLikeOpInterface(op);
  }];
}

