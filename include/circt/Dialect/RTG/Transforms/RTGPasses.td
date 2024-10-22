//===-- RTGPasses.td - RTG pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the passes that work on the RTG dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_RTG_RTGPASSES_TD
#define CIRCT_DIALECT_RTG_RTGPASSES_TD

include "mlir/Pass/PassBase.td"

def ElaborationPass : Pass<"rtg-elaborate", "mlir::ModuleOp"> {
  let summary = "Elaborate the randomizing parts of RTG";
}

def ContextPass : Pass<"rtg-context", "mlir::ModuleOp"> {
  let summary = "Elaborate the contexts of RTG";
}

#endif // CIRCT_DIALECT_RTG_RTGPASSES_TD
