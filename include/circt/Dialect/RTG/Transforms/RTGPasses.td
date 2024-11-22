//===-- RTGPasses.td - RTG pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the passes that operate on the RTG dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_RTG_RTGPASSES_TD
#define CIRCT_DIALECT_RTG_RTGPASSES_TD

include "mlir/Pass/PassBase.td"

def ElaborationPass : Pass<"rtg-elaborate", "mlir::ModuleOp"> {
  let summary = "elaborate the randomization parts";
  let description = [{
    This pass interprets most RTG operations to perform the represented
    randomization and in the process get rid of those operations. This means,
    after this pass the IR does not contain any random constructs within tests
    anymore.
  }];

  // Define a custom constructor to have more control over the pass options
  // (e.g., std::optional options are not handled very well).
  let constructor = "::circt::rtg::createElaborationPass()";
}

#endif // CIRCT_DIALECT_RTG_RTGPASSES_TD
