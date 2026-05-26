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

def SquashSimTriggered : Pass<"sim-squash-triggered", "hw::HWModuleOp"> {
  let summary = "Merge sim.triggered ops with the same clock.";
  let description = [{
    Merge multiple `sim.triggered` operations in the same block that share the
    same clock into a single `sim.triggered`. When the original operations use
    different trigger conditions, those conditions may be materialized as
    `scf.if` operations inside the merged body as needed. This pass does not
    make any guarantees about the relative execution order of the original
    `sim.triggered` operations.
  }];
  let dependentDialects = ["circt::hw::HWDialect, circt::seq::SeqDialect, mlir::scf::SCFDialect"];
  let options = [
    Option<"convertToHW", "convert-to-hw", "bool", "false",
           "Convert squashed sim.triggered ops to hw.triggered">
  ];
}

def LowerDPIFunc : Pass<"sim-lower-dpi-func", "mlir::ModuleOp"> {
  let summary = "Lower sim.dpi.func into func.func for the simulation flow";
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::LLVM::LLVMDialect"];
}

#endif // CIRCT_DIALECT_SIM_SEQPASSES
