//===-- FSMPasses.td - FSM pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FSM_FSMPASSES
#define CIRCT_DIALECT_FSM_FSMPASSES

include "mlir/Pass/PassBase.td"

def PrintStateGraph : Pass<"fsm-print-state-graph", "ModuleOp"> {
  let summary = "Print a DOT graph of the state machine";
  let constructor =  "circt::fsm::createPrintStateGraphPass()";
}

#endif // CIRCT_DIALECT_FSM_FSMPASSES
