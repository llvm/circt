//===-- Passes.td - BMC Transforms pass definition file ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for BMC transformation passes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TOOLS_CIRCT_BMC_PASSES_TD
#define CIRCT_TOOLS_CIRCT_BMC_PASSES_TD

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def LowerToBMC : Pass<"lower-to-bmc", "::mlir::ModuleOp"> {
  let summary = "Lower CIRCTs core dialects to a BMC problem statement";
  let options = [
    Option<"topModule", "top-module", "std::string",
           /*default=*/"",
           "Name of the top module to verify.">,
    Option<"bound", "bound", "unsigned",
           /*default=*/"",
           "Cycle bound.">,
    Option<"ignoreAssertionsUntil", "ignore-asserts-until", "unsigned",
           /*default=*/"0",
           "Specifies an initial window of cycles where assertions should be ignored (starting from 0).">,
    Option<"insertMainFunc", "insert-main", "bool",
           /*default=*/"false",
           "Whether a main function should be inserted for AOT compilation.">,
    Option<"risingClocksOnly", "rising-clocks-only", "bool",
           /*default=*/"false",
           "Only consider the circuit and property on rising clock edges.">,
  ];

  let dependentDialects = [
    "mlir::func::FuncDialect", "mlir::LLVM::LLVMDialect",
    "circt::verif::VerifDialect", "circt::comb::CombDialect",
    "circt::debug::DebugDialect"
  ];
}

def ExternalizeRegisters : Pass<"externalize-registers", "::mlir::ModuleOp"> {
  let summary = "Removes registers and adds corresponding input and output ports";

  let dependentDialects = [
    "circt::seq::SeqDialect", "circt::hw::HWDialect"
  ];
}

def MaterializeDebugVariables : Pass<"materialize-debug-variables", "::mlir::ModuleOp"> {
  let summary = "Materializes debug anchors for module inputs and registers";
  let description = [{
    Creates `dbg.variable` anchors for module inputs and register
    results that do not yet have one.
  }];

  let dependentDialects = [
    "circt::debug::DebugDialect", "circt::seq::SeqDialect",
    "circt::hw::HWDialect"
  ];
}

def PrepareForBMC : Pass<"prepare-for-bmc", "::mlir::ModuleOp"> {
  let summary = "Normalizes clock ports and clocked properties for BMC";
  let description = [{
    Converts `i1` module inputs used through `seq.to_clock` into native
    `!seq.clock` inputs. It also converts clocked assertions and assumptions
    into unclocked properties gated by an actual edge, tracked by comparing the
    current clock value with a registered previous value.
  }];
  let options = [
    Option<"topModule", "top-module", "std::string",
           /*default=*/"",
           "Name of the top module to prepare.">
  ];

  let dependentDialects = [
    "circt::comb::CombDialect", "circt::hw::HWDialect",
    "circt::seq::SeqDialect", "circt::verif::VerifDialect"
  ];
}

#endif // CIRCT_TOOLS_CIRCT_BMC_PASSES_TD
