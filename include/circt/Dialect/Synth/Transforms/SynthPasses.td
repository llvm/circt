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

def LowerVariadic : Pass<"synth-lower-variadic", "hw::HWModuleOp"> {
  let summary = "Lower variadic AndInverter operations to binary AndInverter";
}

def LowerWordToBits : Pass<"synth-lower-word-to-bits", "hw::HWModuleOp"> {
  let summary = "Lower multi-bit AndInverter to single-bit ones";
  let dependentDialects = ["comb::CombDialect"];
}

def PrintLongestPathAnalysis
    : Pass<"synth-print-longest-path-analysis", "mlir::ModuleOp"> {
  let summary =
      "Print longest path analysis results with detailed timing statistics";
  let description = [{
    This pass performs longest path analysis on AIG circuits and outputs detailed
    timing information including:
    - Delay distribution statistics showing timing levels and path counts
    - Critical path details for the top N endPoints
    - Path history with intermediate debug points for detailed analysis

    The analysis considers each AIG and-inverter operation to have unit delay and
    computes maximum delays through combinational paths across module hierarchies.
  }];
  let options = [Option<
                     "outputFile", "output-file", "std::string", "\"-\"",
                     "Output file for analysis results (use '-' for stdout)">,
                 Option<"test", "test", "bool", "false",
                        "Emit longest paths as diagnostic remarks for testing">,
                 Option<"showTopKPercent", "show-top-k-percent", "int", "5",
                        "The size of the longest paths to show.">,
                 Option<"emitJSON", "emit-json", "bool", "false",
                        "Output analysis results in JSON format">];
}

class ExternalSolverPass<string name> : Pass<name, "hw::HWModuleOp"> {
  list<Option> baseOptions = [Option<"continueOnFailure", "continue-on-failure",
                                     "bool", "false",
                                     "Don't fail even if the AIGER exporter, "
                                     "external solver, or AIGER importer fail">,
  ];

  let dependentDialects = ["circt::comb::CombDialect", "circt::hw::HWDialect",
                           "circt::seq::SeqDialect"];
}

def AIGERRunner : ExternalSolverPass<"synth-aiger-runner"> {
  let summary = "Run external solver on AIGER files";
  let description = [{
    This pass runs an external solver on AIGER files. It exports the current
    module to AIGER format, runs the external solver, and imports the result
    back into the module.
  }];
  let options =
      baseOptions#[Option<"aigerPath", "aiger-path", "std::string", "",
                          "Path to the AIGER file">,
                   Option<"solverPath", "solver-path", "std::string", "",
                          "Path to the external solver">,
                   ListOption<"solverArgs", "solver-args", "std::string", "">,
  ];
}

def ABCRunner : ExternalSolverPass<"synth-abc-runner"> {
  let summary = "Run ABC on AIGER files";
  let description = [{
    This pass runs ABC on AIGER files. It is a wrapper around AIGERRunner that
    uses ABC as the external solver. It runs the following ABC commands:
    - `read <inputFile>`: Read the AIGER file
    - for each command in `abcCommands`, run `-q <command>`
    - `write <outputFile>`: Write the AIGER file
  }];
  let options =
      baseOptions#[Option<"abcPath", "abc-path", "std::string", "",
                          "Path to the ABC executable">,
                   ListOption<"abcCommands", "abc-commands", "std::string", "">,
  ];
}

def StructuralHash : Pass<"synth-structural-hash", "hw::HWModuleOp"> {
  let summary = "Structural hashing (CSE) for Synth operations";
  let description = [{
    This pass performs aggressive structural hashing-based CSE for Synth dialect
    operations (AIG/MIG), domain-specific to AIG/MIG operations to enable operand
    reordering based on structural properties and inversion flag consideration
    for canonicalization.
  }];
}

#endif // CIRCT_DIALECT_SYNTH_TRANSFORMS_PASSES_TD
