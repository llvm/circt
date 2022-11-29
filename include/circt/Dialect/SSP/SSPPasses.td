//===- SSPPasses.td - SSP pass definition file -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the passes that work on the SSP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SSP_SSPPASSES_TD
#define CIRCT_DIALECT_SSP_SSPPASSES_TD

include "mlir/Pass/PassBase.td"

def PrintInstance : Pass<"ssp-print-instance", "mlir::ModuleOp"> {
  let summary = "Prints all SSP instances as DOT graphs.";
  let constructor = "circt::ssp::createPrintInstancePass()";
}

#endif // CIRCT_DIALECT_SSP_SSPPASSES_TD
