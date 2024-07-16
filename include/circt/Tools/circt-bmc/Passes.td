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

def ExternalizeRegisters : Pass<"externalize-registers", "::mlir::ModuleOp"> {
  let summary = "Removes registers and adds corresponding input and output ports";

  let dependentDialects = [
    "circt::seq::SeqDialect", "circt::hw::HWDialect"
  ];
}

#endif // CIRCT_TOOLS_CIRCT_BMC_PASSES_TD

