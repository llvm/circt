//===-- Passes.td - LLHD to LLVM pass definition file ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
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
