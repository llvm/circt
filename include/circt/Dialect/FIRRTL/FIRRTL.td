//===- FIRRTL.td - FIRRTL dialect definition ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the FIRRTL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef FIRRTL_TD
#define FIRRTL_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/RegionKindInterface.td"

def FIRRTLDialect : Dialect {
  let name = "firrtl";

  let summary = "Types and operations for firrtl dialect";
  let description = [{
    This dialect defines the `firrtl` dialect, which is used to lower from
    Chisel code to Verilog.  For more information, see the
    [FIRRTL GitHub page](https://github.com/freechipsproject/firrtl).
  }];

  let hasConstantMaterializer = 1;
  let cppNamespace = "::circt::firrtl";
  let extraClassDeclaration = [{
    /// Register all FIRRTL types.
    void registerTypes();
  }];
}

// Base class for the operation in this dialect.
class FIRRTLOp<string mnemonic, list<OpTrait> traits = []> :
    Op<FIRRTLDialect, mnemonic, traits>;

include "FIRRTLTypes.td"
include "FIRRTLStructure.td"
include "FIRRTLDeclarations.td"
include "FIRRTLExpressions.td"
include "FIRRTLStatements.td"

#endif // FIRRTL_TD
