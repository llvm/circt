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

def InlineModules : Pass<"arc-inline-modules", "mlir::ModuleOp"> {
  let summary = "Eagerly inline private modules";
  let description = [{
    This pass eagerly inlines private HW modules into their instantiation sites.
    After outlining combinational logic and registers into arcs, module bodies
    become fairly lightweight. Since arc definitions now fulfill the purpose of
    code reuse by allowing a single definition to be called multiple times, the
    module hierarchy degenerates into a purely cosmetic construct. At that point
    it is beneficial to fully flatten the module hierarchy to simplify further
    analysis and optimization of state transfer arcs.
  }];
  let constructor = "circt::arc::createInlineModulesPass()";
}

def SplitLoops : Pass<"arc-split-loops", "mlir::ModuleOp"> {
  let summary = "Split arcs to break zero latency loops";
  let constructor = "circt::arc::createSplitLoopsPass()";
  let dependentDialects = ["arc::ArcDialect"];
}

#endif // CIRCT_DIALECT_ARC_PASSES_TD
