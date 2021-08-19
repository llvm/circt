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

#ifndef HW_TD
#define HW_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/RegionKindInterface.td"

include "circt/Dialect/HW/HWInterfaces.td"

def HWDialect : Dialect {
  let name = "hw";

  let summary = "Types and operations for the hardware dialect";
  let description = [{
    This dialect defines the `hw` dialect, which is intended to be a generic
    representation of HW outside of a particular use-case.
  }];

  let hasConstantMaterializer = 1;
  let cppNamespace = "::circt::hw";
  let extraClassDeclaration = [{
    /// Register all HW types.
    void registerTypes();
  }];
}

include "circt/Dialect/HW/HWTypes.td"

// Base class for the operation in this dialect.
class HWOp<string mnemonic, list<OpTrait> traits = []> :
    Op<HWDialect, mnemonic, traits>;

include "circt/Dialect/HW/HWMiscOps.td"
include "circt/Dialect/HW/HWAggregates.td"
include "circt/Dialect/HW/HWStructure.td"
include "circt/Dialect/HW/HWTypeDecls.td"

#endif // HW_TD
