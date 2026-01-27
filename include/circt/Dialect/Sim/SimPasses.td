//===-- SimPasses.td - Sim pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for passes that work on the Sim dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SIM_SEQPASSES
#define CIRCT_DIALECT_SIM_SEQPASSES

include "mlir/Pass/PassBase.td"

def FoldValueFormatters : Pass<"sim-fold-value-formatters", "mlir::ModuleOp"> {
  let summary = "Convert constant formatted values to literals.";
}

def ProceduralizeSim : Pass<"sim-proceduralize", "hw::HWModuleOp"> {
  let summary = "Transform non-procedural to procedural operations.";
  let description = [{Transform non-procedural simulation operations with clock
                      and enable to procedural operations wrapped in
                      a procedural region.}];
  let dependentDialects = ["circt::hw::HWDialect, circt::seq::SeqDialect, mlir::scf::SCFDialect"];
}

def LowerDPIFunc : Pass<"sim-lower-dpi-func", "mlir::ModuleOp"> {
  let summary = "Lower sim.dpi.func into func.func for the simulation flow";
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::LLVM::LLVMDialect"];
}

#endif // CIRCT_DIALECT_SIM_SEQPASSES
