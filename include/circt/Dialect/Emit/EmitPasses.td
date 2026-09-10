//===- EmitPasses.td - Emit dialect passes -----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

def StripEmitPass : Pass<"strip-emit", "mlir::ModuleOp"> {
  let summary = "Remove Emit dialect ops";
  let description = [{
    Removes all Emit dialect operations from the IR. Useful as a prepass in
    pipelines that have to discard Emit operations.
  }];
}
