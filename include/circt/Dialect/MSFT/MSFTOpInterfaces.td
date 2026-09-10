//===- MSFTOpInstances.td - MSFT OpInstance definitions ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"

def DynInstDataOpInterface : OpInterface<"DynInstDataOpInterface"> {
  let description = [{
    Interface for ops that define dynamic instance data.
  }];
  let cppNamespace = "::circt::msft";

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Get the top module op to which this op is providing data for.
      }],
      /*retTy=*/"Operation *",
      /*methodName=*/"getTopModule",
      /*args=*/(ins "circt::hw::HWSymbolCache &":$symCache)
    >
  ];
}

  def UnaryDynInstDataOpInterface : OpInterface<"UnaryDynInstDataOpInterface",
    [DynInstDataOpInterface]> {  
  let description = [{
    Interface for anything which needs to refer to a HierPathOp.
  }];
  let cppNamespace = "::circt::msft";
  let verify = [{
    return ::circt::msft::verifyUnaryDynInstDataOp($_op);
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Set the HierPathOp to which this op is referring.
      }],
      /*retTy=*/"void",
      /*methodName=*/"setPathOp",
      /*args=*/(ins "::circt::hw::HierPathOp":$ref),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        $_op.setRefAttr(FlatSymbolRefAttr::get(ref));
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Get the symbol of the HierPathOp to which this op is referring.
      }],
      /*retTy=*/"FlatSymbolRefAttr",
      /*methodName=*/"getPathSym",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op.getRefAttr();
      }]
    >
  ];
}
