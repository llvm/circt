//===-- Passes.td - HW pass definition file ----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the passes that work on the HW dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_PASSES_TD
#define CIRCT_DIALECT_HW_PASSES_TD

include "mlir/Pass/PassBase.td"

def PrintInstanceGraph : Pass<"hw-print-instance-graph", "mlir::ModuleOp"> {
  let summary = "Print a DOT graph of the module hierarchy.";
  let constructor =  "circt::hw::createPrintInstanceGraphPass()";
}

def PrintHWModuleGraph : Pass<"hw-print-module-graph", "mlir::ModuleOp"> {
  let summary = "Print a DOT graph of the HWModule's within a top-level module.";
  let constructor =  "circt::hw::createPrintHWModuleGraphPass()";
  
  let options = [
    Option<"verboseEdges", "verbose-edges", "bool", "false",
      "Print information on SSA edges (types, operand #, ...)">,
  ];
}

def FlattenIO : Pass<"hw-flatten-io", "mlir::ModuleOp"> {
  let summary = "Flattens hw::Structure typed in- and output ports.";
  let constructor =  "circt::hw::createFlattenIOPass()";
  
  let options = [
    Option<"recursive", "recursive", "bool", "false",
      "Recursively flatten nested structs.">,
  ];
}

def HWSpecialize : Pass<"hw-specialize", "mlir::ModuleOp"> {
  let summary = "Specializes instances of parametric hw.modules";
  let constructor = "circt::hw::createHWSpecializePass()";
  let description = [{
    Any `hw.instance` operation instantiating a parametric `hw.module` will
    trigger a specialization procedure which resolves all parametric types and
    values within the module based on the set of provided parameters to the
    `hw.instance` operation. This specialized module is created as a new
    `hw.module` and the referring `hw.instance` operation is rewritten to
    instantiate the newly specialized module.
  }];
}

def InnerSymbolDCE : Pass<"hw-inner-symbol-dce", "mlir::ModuleOp"> {
  let summary = "Eliminate dead inner symbols";
  let description = [{
    This pass deletes all inner symbols which have no uses. This is necessary to
    unblock optimizations and removal of the operations which have these unused
    inner symbols.
  }];
  let constructor = "circt::hw::createInnerSymbolDCEPass()";
  let statistics = [
    Statistic<"numInnerRefsFound", "num-inner-refs-found", "Number of inner-refs found">,
    Statistic<"numInnerSymbolsFound", "num-inner-sym-found", "Number of inner symbols found">,
    Statistic<"numInnerSymbolsRemoved", "num-inner-sym-removed", "Number of inner symbols removed">
  ];
}


#endif // CIRCT_DIALECT_HW_PASSES_TD
