//===- CIRCTStandaloneOps.td - CIRCT standalone ops --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_STANDALONE_OPS
#define CIRCT_STANDALONE_OPS

include "CIRCTStandalone/CIRCTStandaloneDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def CIRCTStandalone_FooOp : CIRCTStandaloneOp<"foo", [Pure]> {
  let summary = "Illustrates how to define an operation";
  let description = [{
    The `circt_standalone.foo` operation illustrates the minimum TableGen
    needed for a custom operation in an out-of-tree CIRCT dialect.
  }];

  let assemblyFormat = "attr-dict";
}

#endif // CIRCT_STANDALONE_OPS
