//===-- Passes.td - FIRRTL to LLHD pass definition file ----*- tablegen -*-===//
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
}

#endif // CIRCT_CONVERSION_FIRRTLTOLLHD_PASSES
