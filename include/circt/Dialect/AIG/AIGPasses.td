//===- AIGPasses.td - AIG dialect passes -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_AIG_AIGPASSES_TD
#define CIRCT_DIALECT_AIG_AIGPASSES_TD

include "mlir/Pass/PassBase.td"

def LowerVariadic : Pass<"aig-lower-variadic", "hw::HWModuleOp"> {
  let summary = "Lower variadic AndInverter operations to binary AndInverter";
}

def LowerWordToBits : Pass<"aig-lower-word-to-bits", "hw::HWModuleOp"> {
  let summary = "Lower multi-bit AndInverter to single-bit ones";
  let dependentDialects = ["comb::CombDialect"];
}

def PrintLongestPathAnalysis : Pass<"aig-print-longest-path-analysis", "mlir::ModuleOp"> {
  let summary = "Print longest path analysis";
  let options = [
    Option<"outputFile", "output-file", "std::string", "\"-\"", "Output file">,
    Option<"test", "test", "bool", "false", "Emit longest paths to the diagnostics.">,
    Option<"showTopKPercent", "show-top-k-percent", "int", "5", "The size of the longest paths to show.">
  ];
}

def AIGERRunner : Pass<"aig-runner", "hw::HWModuleOp"> {
  let summary = "Run external logic solvers on AIGER files";
  let description = [{
    This pass exports the current hardware module to AIGER format, runs an
    external logic solver on the exported file, and imports the optimized
    results back into the MLIR module.

    The solver command arguments must contain placeholder tokens that are
    replaced at runtime:
    - `<inputFile>`: Replaced with the path to the temporary AIGER input file
    - `<outputFile>`: Replaced with the path where the solver should write results.

    ABCRunner is a wrapper around AIGERRunner that uses ABC as the external
    solver. Hence if the user wants to use ABC, they should use ABCRunner instead
    of AIGERRunner.
  }];
  let options = [
    Option<"solverPath", "solver-path", "std::string", "",
           "Path to the external solver executable">,
    ListOption<"solverArgs", "solver-args", "std::string", "">
  ];

  let dependentDialects = [
    "circt::comb::CombDialect", "circt::hw::HWDialect", "circt::seq::SeqDialect"
  ];
}

def ABCRunner : Pass<"abc-runner", "hw::HWModuleOp"> {
  let summary = "Run ABC on AIGER files";
  let description = [{
    This pass runs ABC on AIGER files. It is a wrapper around AIGERRunner that
    uses ABC as the external solver. It runs the following ABC commands:
    - `read <inputFile>`: Read the AIGER file
    - for each command in `abcCommands`, run `-q <command>`
    - `write <outputFile>`: Write the AIGER file
  }];
  let options = [
    Option<"abcPath", "abc-path", "std::string", "", "Path to the ABC executable">,
    ListOption<"abcCommands", "abc-commands", "std::string", "">
  ];

  let dependentDialects = [
    "circt::comb::CombDialect", "circt::hw::HWDialect", "circt::seq::SeqDialect"
  ];
}

#endif // CIRCT_DIALECT_AIG_AIGPASSES_TD
