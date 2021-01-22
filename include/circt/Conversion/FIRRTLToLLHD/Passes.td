//===-- Passes.td - FIRRTL to LLHD pass definition file ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for the FIRRTL to LLHD conversion pass.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_FIRRTLTOLLHD_PASSES
#define CIRCT_CONVERSION_FIRRTLTOLLHD_PASSES

include "mlir/Pass/PassBase.td"

def ConvertFIRRTLToLLHD : Pass<"convert-firrtl-to-llhd", "ModuleOp"> {
  let summary = "Convert FIRRTL to LLHD";
  let description = [{
    This pass translates a FIRRTL design into an equivalent structural LLHD
    description.
  }];
  let constructor = "circt::llhd::createConvertFIRRTLToLLHDPass()";
  let dependentDialects = ["llhd::LLHDDialect"];
}

#endif // CIRCT_CONVERSION_FIRRTLTOLLHD_PASSES
