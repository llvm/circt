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

def AddTaps : Pass<"arc-add-taps", "mlir::ModuleOp"> {
  let summary = "Add taps to ports and wires such that they remain observable";
  let constructor = "circt::arc::createAddTapsPass()";
  let dependentDialects = ["arc::ArcDialect"];
  let options = [
    Option<"tapPorts", "ports", "bool", "true", "Make module ports observable">,
    Option<"tapWires", "wires", "bool", "true", "Make wires observable">
  ];
}

def AllocateState : Pass<"arc-allocate-state", "arc::ModelOp"> {
  let summary = "Allocate and layout the global simulation state";
  let constructor = "circt::arc::createAllocateStatePass()";
  let dependentDialects = ["arc::ArcDialect"];
}

def Dedup : Pass<"arc-dedup", "mlir::ModuleOp"> {
  let summary = "Deduplicate identical arc definitions";
  let description = [{
    This pass deduplicates identical arc definitions. If two arcs differ only by
    constants, the constants are outlined such that the arc can be deduplicated.
  }];
  let constructor = "circt::arc::createDedupPass()";
  let dependentDialects = ["arc::ArcDialect"];
}

def InferMemories : Pass<"arc-infer-memories", "mlir::ModuleOp"> {
  let summary = "Convert `FIRRTL_Memory` instances to dedicated memory ops";
  let constructor = "circt::arc::createInferMemoriesPass()";
  let dependentDialects = [
    "arc::ArcDialect", "comb::CombDialect", "seq::SeqDialect"
  ];
}

def InlineArcs : Pass<"arc-inline" , "mlir::ModuleOp"> {
  let summary = "Inline very small arcs";
  let constructor = "circt::arc::createInlineArcsPass()";
  let statistics = [
    Statistic<"numInlinedArcs", "inlined-arcs", "Arcs inlined at a use site">,
    Statistic<"numRemovedArcs", "removed-arcs",
      "Arcs removed after full inlining">,
    Statistic<"numTrivialArcs", "trivial-arcs", "Arcs with very few ops">,
    Statistic<"numSingleUseArcs", "single-use-arcs", "Arcs with a single use">,
  ];
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

def InferStateProperties : Pass<"arc-infer-state-properties",
                                "mlir::ModuleOp"> {
  let summary = "Add resets and enables explicitly to the state operations";
  let constructor = "circt::arc::createInferStatePropertiesPass()";
  let dependentDialects = ["circt::hw::HWDialect", "circt::comb::CombDialect"];
}

def LegalizeStateUpdate : Pass<"arc-legalize-state-update", "mlir::ModuleOp"> {
  let summary = "Insert temporaries such that state reads don't see writes";
  let constructor = "circt::arc::createLegalizeStateUpdatePass()";
  let dependentDialects = ["arc::ArcDialect"];
}

def LowerClocksToFuncs : Pass<"arc-lower-clocks-to-funcs", "mlir::ModuleOp"> {
  let summary = "Lower clock trees into functions";
  let constructor = "circt::arc::createLowerClocksToFuncsPass()";
  let dependentDialects = ["mlir::func::FuncDialect"];
}

def LowerLUT : Pass<"arc-lower-lut", "arc::DefineOp"> {
  let summary = "Lowers arc.lut into a comb and hw only representation.";
  let constructor = "circt::arc::createLowerLUTPass()";
  let dependentDialects = ["hw::HWDialect", "comb::CombDialect"];
}

def LowerState : Pass<"arc-lower-state", "mlir::ModuleOp"> {
  let summary = "Split state into read and write ops grouped by clock tree";
  let constructor = "circt::arc::createLowerStatePass()";
  let dependentDialects = ["arc::ArcDialect", "mlir::scf::SCFDialect"];
}

def MakeTables : Pass<"arc-make-tables", "mlir::ModuleOp"> {
  let summary = "Transform appropriate arc logic into lookup tables";
  let constructor = "circt::arc::createMakeTablesPass()";
  let dependentDialects = ["arc::ArcDialect"];
}

def MuxToControlFlow : Pass<"arc-mux-to-control-flow", "mlir::ModuleOp"> {
  let summary = "Convert muxes with large independent fan-ins to if-statements";
  let constructor = "circt::arc::createMuxToControlFlowPass()";
  let dependentDialects = ["mlir::scf::SCFDialect"];
}

def PrintStateInfo : Pass<"arc-print-state-info", "mlir::ModuleOp"> {
  let summary = "Print the state storage layout in JSON format";
  let constructor = "circt::arc::createPrintStateInfoPass()";
  let options = [
    Option<"stateFile", "state-file", "std::string", "",
      "Emit file with state description">
  ];
}

def RemoveUnusedArcArguments : Pass<"arc-remove-unused-arc-arguments",
    "mlir::ModuleOp"> {
  let summary =
    "Remove unused arc args from the arc itself and the referencing states";
  let constructor = "circt::arc::createRemoveUnusedArcArgumentsPass()";
}

def SimplifyVariadicOps : Pass<"arc-simplify-variadic-ops", "mlir::ModuleOp"> {
  let summary = "Convert variadic ops into distributed binary ops";
  let constructor = "circt::arc::createSimplifyVariadicOpsPass()";
  let statistics = [
    Statistic<"numOpsSkippedMultipleBlocks", "skipped-multiple-blocks",
      "Ops skipped due to operands in different blocks">,
    Statistic<"numOpsSimplified", "simplified",
      "Ops simplified into binary ops">,
    Statistic<"numOpsCreated", "created",
      "Ops created as part of simplification">,
    Statistic<"numOpsReordered", "reordered",
      "Ops where simplification reordered operands">,
  ];
}

def SinkInputs : Pass<"arc-sink-inputs", "mlir::ModuleOp"> {
  let summary = "Sink constant inputs into arcs";
  let constructor = "circt::arc::createSinkInputsPass()";
  let dependentDialects = ["arc::ArcDialect"];
}

def SplitLoops : Pass<"arc-split-loops", "mlir::ModuleOp"> {
  let summary = "Split arcs to break zero latency loops";
  let constructor = "circt::arc::createSplitLoopsPass()";
  let dependentDialects = ["arc::ArcDialect"];
}

def StripSV : Pass<"arc-strip-sv", "mlir::ModuleOp"> {
  let summary = "Remove SV wire, reg, and assigns";
  let constructor = "circt::arc::createStripSVPass()";
  let dependentDialects = ["seq::SeqDialect", "arc::ArcDialect"];
}

#endif // CIRCT_DIALECT_ARC_PASSES_TD
