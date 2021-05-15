//===- HW.td - HW dialect definition -----------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the HW dialect.
//
//===----------------------------------------------------------------------===//

#ifndef RTL_TD
#define RTL_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/RegionKindInterface.td"

def RTLDialect : Dialect {
  let name = "rtl";

  let summary = "Types and operations for rtl dialect";
  let description = [{
    This dialect defines the `rtl` dialect, which is intended to be a generic
    representation of RTL outside of a particular use-case.
  }];

  let hasConstantMaterializer = 1;
  let cppNamespace = "::circt::rtl";
  let extraClassDeclaration = [{
    /// Register all RTL types.
    void registerTypes();
  }];
}

include "circt/Dialect/HW/HWTypes.td"

// Base class for the operation in this dialect.
class RTLOp<string mnemonic, list<OpTrait> traits = []> :
    Op<RTLDialect, mnemonic, traits>;

include "circt/Dialect/HW/HWAggregates.td"
include "circt/Dialect/HW/HWStructure.td"
include "circt/Dialect/HW/HWTypeDecls.td"

#endif // RTL_TD
