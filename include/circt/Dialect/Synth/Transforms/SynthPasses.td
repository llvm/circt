//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for passes related to Synthesis.
//
//===----------------------------------------------------------------------===//


#ifndef CIRCT_DIALECT_SYNTH_TRANSFORMS_PASSES_TD
#define CIRCT_DIALECT_SYNTH_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def TestPriorityCuts : Pass<"synth-test-priority-cuts", "hw::HWModuleOp"> {
  let summary = "Test priority cuts for synthesis";
  let options = [
    Option<"maxCutsPerRoot", "max-cuts-per-root", "int", "6",
           "Maximum number of cuts to maintain per node">,
    Option<"maxCutInputSize", "max-cut-input-size", "int", "6",
           "Maximum number of cut inputs to consider">,
  ];
}

#endif // CIRCT_DIALECT_SYNTH_TRANSFORMS_PASSES_TD
