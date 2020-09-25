//===-- Passes.td - LLHD to LLVM pass definition file ------*- tablegen -*-===//
//
// This file contains definitions for the LLHD to LLVM conversion pass.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_LLHDTOLLVM_PASSES
#define CIRCT_CONVERSION_LLHDTOLLVM_PASSES

include "mlir/Pass/PassBase.td"

def ConvertLLHDToLLVM : Pass<"convert-llhd-to-llvm", "ModuleOp"> {
    let summary = "Convert LLHD to LLVM";
    // TODO: add description
    let description = [{
        TODO
    }];

    let constructor = "circt::llhd::createConvertLLHDToLLVMPass()";
}

#endif // CIRCT_CONVERSION_LLHDTOLLVM_PASSES
