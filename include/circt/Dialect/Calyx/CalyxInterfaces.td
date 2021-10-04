//===- CalyxInterfaces.td - Calyx Interfaces -------------*- tablegen -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the interfaces in the Calyx dialect.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"

// Op interface for groups.
def GroupOpInterface : OpInterface<"GroupInterface"> {
  let cppNamespace = "::circt::calyx";

  let description = [{
    This is an op interface for Calyx Groups.
  }];

  let methods = [
    InterfaceMethod<
      "This returns the symbol name of the group.",
      "mlir::StringAttr",
      "symName",
      (ins), [{
        Operation* op = $_op;
        return op->getAttrOfType<mlir::StringAttr>("sym_name");
      }]>,
    InterfaceMethod<
      "This returns the body of the group.",
      "Block*",
      "getBody",
      (ins), [{
        Region* region = &$_op->getRegion(0);
        assert(region->hasOneBlock() && "The body should have one Block.");
        return &region->front();
      }]>
  ];
  let verify = [{ return verifyGroupInterface(op); }];
}

/// Op Interface for cells.
def CellOpInterface : OpInterface<"CellInterface"> {
  let cppNamespace = "::circt::calyx";

  let description = [{
    This is an op interface for Calyx Cells. Cells consist
    of primitives and instances of components.
  }];

  let methods = [
    InterfaceMethod<
      "This returns the port names associated with the cell.",
      "SmallVector<StringRef>", "portNames"
    >,
    InterfaceMethod<
      "This returns the port directions associated with the cell.",
      "SmallVector<circt::calyx::Direction>", "portDirections"
    >,
    InterfaceMethod<
      "This returns the port attributes associated with the cell.",
      "SmallVector<DictionaryAttr>", "portAttributes"
    >,
    InterfaceMethod<
      "This returns the instance name of the cell.",
      "StringRef",
      "instanceName",
      (ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        Operation* op = (*static_cast<ConcreteOp *>($_op));
        return op->getAttrOfType<mlir::FlatSymbolRefAttr>("instanceName").getValue();
      }]
    >,
    InterfaceMethod<
      "This returns the index of the port within the result values of this cell.",
      "unsigned",
      "portIndex",
      (ins "Value":$port),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        for (auto it : llvm::enumerate($_op->getResults()))
          if (it.value() == port)
            return it.index();
        assert(false && "argument port not found in this cell.");
        return  {};
      }]
    >,
    InterfaceMethod<
      "This returns the PortInfo associated with all of the ports of a cell.",
      "SmallVector<circt::calyx::PortInfo>",
      "getPortInfo",
      (ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        SmallVector<circt::calyx::PortInfo> ports;
        MLIRContext* context = $_op->getContext();
        auto zippedPortInfo = llvm::zip(
          $_op->getResults(),
          $_op.portDirections(),
          $_op.portNames(),
          $_op.portAttributes()
        );
        for (auto&& [result, direction, name, attributes] : zippedPortInfo)
          ports.push_back(PortInfo{
              StringAttr::get(context, name),
              result.getType(),
              direction,
              attributes
          });
        return ports;
    }]
    >,
    InterfaceMethod<
      "This returns the PortInfo associated with the port of a cell.",
      "circt::calyx::PortInfo",
      "portInfo",
      (ins "Value":$port),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ return getPortInfo()[portIndex(port)]; }]
    >,
    InterfaceMethod<
      "This returns the direction of a given port of a cell.",
      "circt::calyx::Direction",
      "direction",
      (ins "Value":$port),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ return portInfo(port).direction; }]
    >,
    InterfaceMethod<
      "This returns the name of a given port of a cell.",
      "StringRef",
      "portName",
      (ins "Value":$port),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ return portInfo(port).name.getValue(); }]
    >,
    InterfaceMethod<
      "This returns the attributes of a given port of a cell.",
      "DictionaryAttr",
      "portDictionaryAttr",
      (ins "Value":$port),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ return portInfo(port).attributes; }]
    >,
    InterfaceMethod<
      "This returns the input ports of a cell.",
      "SmallVector<Value>",
      "getInputPorts",
      (ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        SmallVector<Value> ports;
        for (auto&& [result, direction] : llvm::zip($_op->getResults(), $_op.portDirections()))
          if (direction == calyx::Direction::Input)
            ports.push_back(result);
        return ports;
      }]
    >,
    InterfaceMethod<
      "This returns the output ports of a cell.",
      "SmallVector<Value>",
      "getOutputPorts",
      (ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        SmallVector<Value> ports;
        for (auto&& [result, direction] : llvm::zip($_op->getResults(), $_op.portDirections()))
          if (direction == calyx::Direction::Output)
            ports.push_back(result);
        return ports;
      }]
    >,
    InterfaceMethod<
      "This returns the ports of a cell, filtering out any interface ports.",
      "SmallVector<Value>",
      "filterInterfacePorts",
      (ins "calyx::Direction":$dir),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        auto isInterfacePort = [](const PortInfo& port) {
          return port.hasAttribute("go")   || port.hasAttribute("reset") ||
                 port.hasAttribute("done") || port.hasAttribute("clk");
        };

        SmallVector<Value> filteredPorts;
        for (auto&& [result, port] : llvm::zip($_op->getResults(), $_op.getPortInfo())) {
          if (port.direction != dir || !isInterfacePort(port))
            continue;
          filteredPorts.push_back(result);
        }
        return filteredPorts;
      }]
    >
  ];

  let verify = [{ return verifyCell(op); }];
}
