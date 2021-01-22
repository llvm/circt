//===-- Passes.td - RTL to LLHD pass definition file ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for the RTL to LLHD conversion pass.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_RTLTOLLHD_PASSES
#define CIRCT_CONVERSION_RTLTOLLHD_PASSES

include "mlir/Pass/PassBase.td"

def ConvertRTLToLLHD : Pass<"convert-rtl-to-llhd", "mlir::ModuleOp"> {
  let summary = "Convert RTL to LLHD";
  let description = [{
    This pass translates a RTL design into an equivalent structural LLHD
    description.
  }];
  let constructor = "circt::llhd::createConvertRTLToLLHDPass()";
  let dependentDialects = ["llhd::LLHDDialect"];
}

#endif // CIRCT_CONVERSION_RTLTOLLHD_PASSES
