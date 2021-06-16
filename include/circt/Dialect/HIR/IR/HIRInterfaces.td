#ifndef HIR_INTERFACES
#define HIR_INTERFACES

include "mlir/IR/OpBase.td"

def CombBinOpInterface: OpInterface<"CombBinOpInterface">{
  let description = [{Interface for HIR binary ops with zero delay such as 
    hir.add, hir.subtract etc.
  }];
  let methods = [
    InterfaceMethod<[{Returns left operand}], "Value","left">,
    InterfaceMethod<[{Returns left operand}], "Value","right">
  ];
}

#endif
