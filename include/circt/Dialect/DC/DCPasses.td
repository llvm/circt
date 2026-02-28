//===- Passes.td - DC dialect passes ----------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_DC_PASSES_TD
#define CIRCT_DIALECT_DC_PASSES_TD

include "mlir/Pass/PassBase.td"

def DCMaterializeForksSinks : Pass<"dc-materialize-forks-sinks"> {
  let summary = "Materialize fork and sink operations.";
  let description = [{
    This pass analyses a function-like operation and inserts fork and sink
    operations ensuring that all values have exactly one use.
  }];
  let constructor = "circt::dc::createDCMaterializeForksSinksPass()";
  let dependentDialects = ["dc::DCDialect"];
}

def DCDematerializeForksSinks : Pass<"dc-dematerialize-forks-sinks"> {
  let summary = "Dematerialize fork and sink operations.";
  let description = [{
    This pass analyses a function-like operation and removes all fork and sink
    operations.
  }];
  let constructor = "circt::dc::createDCDematerializeForksSinksPass()";
  let dependentDialects = ["dc::DCDialect"];
}

def DCDotPrint : Pass<"dc-print-dot", "mlir::ModuleOp"> {
  let summary = "Print .dot graph of a DC function.";
  let description = [{
    This pass analyses a dc modulen and prints a .dot graph of the
    structure. If multiple functions are present in the IR, the top level
    function will be printed, and called functions will be subgraphs within
    the main graph.
  }];
  let constructor = "circt::dc::createDCDotPrintPass()";
}

#endif // CIRCT_DIALECT_DC_PASSES_TD
