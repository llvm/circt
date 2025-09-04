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

class CutRewriterPassBase<string name, string op> : Pass<name, op> {
  list<Option> baseOptions = [
    Option<"maxCutsPerRoot", "max-cuts-per-root", "int", "6",
           "Maximum number of cuts to maintain per node">,
    Option<"strategy", "strategy", "synth::OptimizationStrategy",
           /*default=*/"synth::OptimizationStrategyTiming",
           "Optimization strategy (area vs. timing)",
           [{::llvm::cl::values(
             clEnumValN(synth::OptimizationStrategyArea, "area",
                        "Optimize for area"),
             clEnumValN(synth::OptimizationStrategyTiming, "timing",
                        "Optimize for timing")
           )}]>,
    Option<"test", "test", "bool", "false", "Attach timing to IR for testing">
  ];
}

def TechMapper : CutRewriterPassBase<"synth-tech-mapper", "mlir::ModuleOp"> {
  let summary = "Technology mapping using cut rewriting";
  let description = [{
    This pass performs technology mapping by converting logic network
    (AIG etc) representations into technology-specific gate implementations.
    It uses cut-based rewriting with priority cuts and NPN canonical forms for
    efficient pattern matching.

    The pass serves dual purposes: providing practical technology mapping
    capabilities and acting as a test vehicle for the cut rewriting framework,
    since testing cut enumeration and pattern matching algorithms directly
    would otherwise be difficult without a concrete application.

    Supports both area and timing optimization strategies.
  }];
  let options = baseOptions;
  let dependentDialects = ["hw::HWDialect"];
}

def GenericLutMapper : CutRewriterPassBase<"synth-generic-lut-mapper",
                                           "hw::HWModuleOp"> {
  let summary = "LUT mapping using generic K-input LUTs";
  let description = [{
    This pass performs technology mapping using generic K-input lookup tables
    (LUTs). It converts combinational logic networks into implementations
    using K-input LUTs (comb.truth_table) with unit area cost and delay.
  }];
  let options = baseOptions # [
    Option<"maxLutSize", "max-lut-size", "unsigned", /*default=*/"6",
           "Maximum number of inputs per LUT">
  ];
  let dependentDialects = ["comb::CombDialect"];
}


#endif // CIRCT_DIALECT_SYNTH_TRANSFORMS_PASSES_TD
