//===-- Passes.td - RTL to LLHD pass definition file -------*- tablegen -*-===//
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
}

#endif // CIRCT_CONVERSION_RTLTOLLHD_PASSES
