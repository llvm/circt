//===- SimOpInterfaces.td - Operation Interfaces ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the HW operation interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SIM_SIMOPINTERFACES
#define CIRCT_DIALECT_SIM_SIMOPINTERFACES

include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/OpBase.td"
include "circt/Dialect/Emit/EmitOpInterfaces.td"
include "circt/Support/InstanceGraphInterface.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

def ProceduralRegion : NativeOpTrait<"ProceduralRegion"> {
  let cppNamespace = "::circt::sim";
}

def ProceduralOp : NativeOpTrait<"ProceduralOp"> {
  let cppNamespace = "::circt::sim";
}

def NonProceduralOp : NativeOpTrait<"NonProceduralOp"> {
  let cppNamespace = "::circt::sim";
}

#endif
