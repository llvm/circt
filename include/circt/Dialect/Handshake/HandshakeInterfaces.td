//===- HandshakeInterfaces.td - Handshake interfaces -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for the structured interface for Handshake ops.
//
//===----------------------------------------------------------------------===//

#ifdef HANDSHAKE_INTERFACES
#else
#define HANDSHAKE_INTERFACES

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

def MergeLikeOpInterface : OpInterface<"MergeLikeOpInterface"> {
  let description = [{
     Some handshake operations can have predecessors in other
     blocks. This is primarily useful for verification purposes during
     lowering from other dialect, such as the standard CDFG dialect.
  }];

  let methods = [
    InterfaceMethod<[{
        Returns an operand range over the data signals being merged.
      }],
      "OperandRange", "dataOperands", (ins)
    >,
  ];

  let verify = [{
    auto concreteOp = cast<ConcreteOpType>($_op);

    auto operands = concreteOp.dataOperands();

    if (!operands.size())
      return concreteOp.emitOpError("must have at least one data operand");

    Type resultType = $_op->getResult(0).getType();

    for (auto operand : operands)
      if (operand.getType() != resultType)
        return concreteOp.emitOpError("operand has type ") << operand.getType()
            << ", but result has type " << resultType;

    return success();
  }];
}


#endif // HANDSHAKE_INTERFACES
