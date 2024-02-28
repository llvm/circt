//===- EmitOpInterfaces.td - Operation Interfaces ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the Emit operation interfaces and traits.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_EMIT_EMITOPINTERFACES
#define CIRCT_DIALECT_EMIT_EMITOPINTERFACES

include "mlir/IR/OpBase.td"

def Emittable : NativeOpTrait<"Emittable"> {
  let cppNamespace = "::circt::emit";
}

#endif // CIRCT_DIALECT_EMIT_EMITOPINTERFACES
