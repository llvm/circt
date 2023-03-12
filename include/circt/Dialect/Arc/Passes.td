//===- Passes.td - Arc dialect passes ----------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_PASSES_TD
#define CIRCT_DIALECT_ARC_PASSES_TD

include "mlir/Pass/PassBase.td"

def Dedup : Pass<"arc-dedup", "mlir::ModuleOp"> {
  let summary = "Deduplicate identical arc definitions";
  let description = [{
    This pass deduplicates identical arc definitions. If two arcs differ only by
    constants, the constants are outlined such that the arc can be deduplicated.
  }];
  let constructor = "circt::arc::createDedupPass()";
  let dependentDialects = ["arc::ArcDialect"];
}

#endif // CIRCT_DIALECT_ARC_PASSES_TD
