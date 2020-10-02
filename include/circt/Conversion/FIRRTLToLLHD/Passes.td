//===-- Passes.td - FIRRTL to LLHD pass definition file ----*- tablegen -*-===//
//
// This file contains definitions for the FIRRTL to LLHD conversion pass.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_FIRRTLTOLLHD_PASSES
#define CIRCT_CONVERSION_FIRRTLTOLLHD_PASSES

include "mlir/Pass/PassBase.td"

def ConvertFIRRTLToLLHD : Pass<"convert-firrtl-to-llhd", "ModuleOp"> {
    let summary = "Convert FIRRTL to Structural LLHD";
    // TODO: add description
    let description = [{
        TODO
    }];

    let constructor = "circt::llhd::createConvertFIRRTLToLLHDPass()";
}

#endif // CIRCT_CONVERSION_FIRRTLTOLLHD_PASSES
