//===-- Passes.td - CHALK pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the passes that work on the CHALK dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_CHALK_PASSES_TD
#define CIRCT_DIALECT_CHALK_PASSES_TD

include "mlir/Pass/PassBase.td"

def InitPlacement : Pass<"chalk-init-placement", "chalk::CellOp"> {
  let summary = "Initial Placement of CHALK cells";
  let description = [{
    Provide initial placement of CHALK cells, ensuring there is no overlap.
  }];
  let constructor = "circt::chalk::createInitPlacementPass()";
}

#endif // CIRCT_DIALECT_CHALK_PASSES_TD
