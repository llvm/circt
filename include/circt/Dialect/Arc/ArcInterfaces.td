//===- ArcInterfaces.td - Interfaces used in Arc -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_ARCINTERFACES_TD
#define CIRCT_DIALECT_ARC_ARCINTERFACES_TD

include "mlir/IR/OpBase.td"

def CallOpMutableInterface : OpInterface<"CallOpMutableInterface"> {
  let description = [{
    Can be implemented in addition to `CallOpInterface` to allow mutation of the
    call operation.
  }];
  let cppNamespace = "::circt::arc";

  let methods = [
    InterfaceMethod<[{
        Returns the operands within this call that are used as arguments to the
        callee as a mutable range.
      }],
      "::mlir::MutableOperandRange", "getArgOperandsMutable"
    >,
  ];
}

#endif // CIRCT_DIALECT_ARC_ARCINTERFACES_TD
