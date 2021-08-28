include "mlir/IR/OpBase.td"

def FModuleLike : OpInterface<"FModuleLike"> {
  let cppNamespace = "circt::firrtl";
  let description = [{ Provide common  module information. }];
  let methods = [
    InterfaceMethod<[{Get the module name} }], 
    "StringRef", "moduleName", (ins), 
    /*methodBody=*/[{ return $_op.getName(); }]>,
    
    InterfaceMethod<[{Get the module type} }], 
    "FunctionType", "moduleType", (ins), [{}],
    /*methodBody=*/[{ return $_op.getType().template cast<FunctionType>(); }]>,
    
    InterfaceMethod<[{Get the module port names}], 
    "ArrayAttr", "portNames", (ins), 
    /*methodBody=*/[{return $_op.portNames(); }]>,

    InterfaceMethod<[{Get a module port type}], 
    "FIRRTLType", "portType", (ins "size_t":$portIndex), 
    /*methodBody=*/[{return $_op.getType(). template cast<FunctionType>().getInputs()[portIndex].template cast<FIRRTLType>();; }]>,

    InterfaceMethod<[{Get a module port name}], 
    "StringAttr", "portName", (ins "size_t":$portIndex), 
    /*methodBody=*/[{return $_op.portNames()[portIndex].template cast<StringAttr>(); }]>,

    InterfaceMethod<[{Get infromation about all ports}],
    "SmallVector<ModulePortInfo>", "getPorts">,

    InterfaceMethod<[{Get the module port directions}],
    "IntegerAttr", "getPortDirections", (ins), [{}],
    /*methodBody=*/[{ return $_op->template getAttrOfType<mlir::IntegerAttr>(direction::attrKey); }]>,

    InterfaceMethod<[{Get a module port direction}],
    "Direction", "getPortDirection", (ins "size_t":$portIndex), [{}],
    /*methodBody=*/[{ return direction::get($_op.getPortDirections().getValue()[portIndex]); }]>

  ];
}
