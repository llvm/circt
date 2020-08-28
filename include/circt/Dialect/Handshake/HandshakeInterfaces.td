//===- HandshakeInterfaces.td - Handshake interfaces -------*- tablegen -*-===//
//
// Copyright 2019 The CIRCT Authors.
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// =============================================================================
//
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

	}],
      "OperandRange", "dataOperands", (ins)
    >,
 ];
}


#endif // HANDSHAKE_INTERFACES
