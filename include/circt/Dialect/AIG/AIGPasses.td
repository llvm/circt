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

def PrintResourceUsageAnalysis : Pass<"aig-print-resource-usage-analysis", "mlir::ModuleOp"> {
  let summary = "Print resource usage of AIG operations";
  let options = [
    Option<"outputJSONFile", "json-file", "std::string", "", "Output JSON file">,
    Option<"topModuleName", "top-name", "std::string", "", "Top module name">,
    Option<"printSummary", "print-summary", "bool", "true", "Print summary in stderr">
  ];
}

def PrintLongestPathAnalysis : Pass<"aig-print-longest-path-analysis", "mlir::ModuleOp"> {
  let summary = "Print longest path analysis of AIG operations";
  let options = [
    Option<"outputJSONFile", "json-file", "std::string", "", "Output JSON file">,
    Option<"topModuleName", "top-name", "std::string", "", "Top module name">,
    Option<"test", "test", "bool", "false", "Emit longest paths to the diagnostics.">
    Option<"showSize", "show-size", "int", "1000", "The size of the longest paths to show.">
  ];
}

#endif // CIRCT_DIALECT_AIG_AIGPASSES_TD
