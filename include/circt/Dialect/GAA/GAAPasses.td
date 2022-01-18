//===-- GAAPasses.td - GAA pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for passes that work on the GAA dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_GAA_GAAPASSES_TD
#define CIRCT_DIALECT_GAA_GAAPASSES_TD

include "mlir/Pass/PassBase.td"

def ReferRules : Pass<"gaa-refer-rules", "gaa::CircuitOp"> {
  let summary = "generate all GAA::RuleRefOp for a circuit.";
  let description = [{
    This pass DFS all Rules via InstanceGraph, find all available rules
    to construct gaa::RuleRefOp.
  }];
  let constructor = "circt::gaa::createReferRules()";
}

def GenerateConflictMatrix : Pass<"gaa-generate-conflict-matrix", "gaa::CircuitOp"> {
  let summary = "generate the conflict matrix for scheduling.";
  let description = [{
    This pass use CallGraph to generate the native call stack store conflict
    matrix to CircuitOp, scheduling pragma will be gather as well.
  }];
  let constructor = "circt::gaa::createGenerateConflictMatrix()";
}

#endif // CIRCT_DIALECT_GAA_GAAPASSES_TD
