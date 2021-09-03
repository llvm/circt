include "mlir/IR/OpBase.td"

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
      "This returns the instance name of the cell.",
      "StringRef",
      "instanceName",
      (ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        Operation* op = (*static_cast<ConcreteOp *>($_op));
        return op->getAttrOfType<mlir::StringAttr>("instanceName").getValue();
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
      "This returns the ComponentPortInfo associated with all of the ports of a cell.",
      "SmallVector<circt::calyx::ComponentPortInfo>",
      "portInfos",
      (ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        SmallVector<circt::calyx::ComponentPortInfo> info;
        for (auto it : llvm::zip($_op->getResults(), $_op.portDirections(),$_op.portNames()))
          info.push_back(ComponentPortInfo{StringAttr::get($_op->getContext(),
            std::get<2>(it)), std::get<0>(it).getType(), std::get<1>(it)});
        return info;
    }]
    >,
    InterfaceMethod<
      "This returns the ComponentPortInfo associated with the port of a cell.",
      "circt::calyx::ComponentPortInfo",
      "portInfo",
      (ins "Value":$port),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ return portInfos()[portIndex(port)]; }]
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
      "This returns the input ports of a cell.",
      "SmallVector<Value>",
      "inputPorts",
      (ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        SmallVector<Value> ports;
        for (auto it : llvm::zip($_op->getResults(), $_op.portDirections()))
          if (std::get<1>(it) == calyx::Direction::Input)
            ports.push_back(std::get<0>(it));
        return ports;
      }]
    >,
    InterfaceMethod<
      "This returns the output ports of a cell.",
      "SmallVector<Value>",
      "outputPorts",
      (ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        SmallVector<Value> ports;
        for (auto it : llvm::zip($_op->getResults(), $_op.portDirections()))
          if (std::get<1>(it) == calyx::Direction::Output)
            ports.push_back(std::get<0>(it));
        return ports;
      }]
    >
  ];

  let verify = [{
    return verifyCell(op);
  }];
}
