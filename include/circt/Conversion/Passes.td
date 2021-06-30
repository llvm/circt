//===-- Passes.td - Conversion pass definitions ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for all dialect conversions.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_PASSES_TD
#define CIRCT_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// StandardToHIR
//===----------------------------------------------------------------------===//

def ConvertStandardToHIR : Pass<"convert-to-hir", "mlir::ModuleOp"> {
  let summary = "Convert standard + scf + memref to HIR";
  let description = [{
    Lower a module of standard + scf + memref dialect to the HIR dialect.
  }];

  let constructor = "circt::createConvertStandardToHIRPass()";
  let dependentDialects = ["hir::HIRDialect", "mlir::StandardOpsDialect"];
}

//===----------------------------------------------------------------------===//
// FIRRTLToHW
//===----------------------------------------------------------------------===//

def LowerFIRRTLToHW : Pass<"lower-firrtl-to-hw", "mlir::ModuleOp"> {
  let summary = "Lower FIRRTL to HW";
  let description = [{
    Lower a module of FIRRTL dialect to the HW dialect family.
  }];
  let constructor = "circt::createLowerFIRRTLToHWPass()";
  let dependentDialects = ["comb::CombDialect", "hw::HWDialect",
                           "sv::SVDialect"];
}

//===----------------------------------------------------------------------===//
// HandshakeToFIRRTL
//===----------------------------------------------------------------------===//

def HandshakeToFIRRTL : Pass<"lower-handshake-to-firrtl", "mlir::ModuleOp"> {
  let summary = "Lower Handshake to FIRRTL";
  let description = [{
    Lower Handshake to FIRRTL.
  }];
  let constructor = "circt::createHandshakeToFIRRTLPass()";
  let dependentDialects = ["firrtl::FIRRTLDialect"];
}

//===----------------------------------------------------------------------===//
// LLHDToLLVM
//===----------------------------------------------------------------------===//

def ConvertLLHDToLLVM : Pass<"convert-llhd-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert LLHD to LLVM";
    let description = [{
        This pass translates LLHD to LLVM. 
    }];

    let constructor = "circt::createConvertLLHDToLLVMPass()";
    let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}

//===----------------------------------------------------------------------===//
// HWToLLHD
//===----------------------------------------------------------------------===//

def ConvertHWToLLHD : Pass<"convert-hw-to-llhd", "mlir::ModuleOp"> {
  let summary = "Convert HW to LLHD";
  let description = [{
    This pass translates a HW design into an equivalent structural LLHD
    description.
  }];
  let constructor = "circt::createConvertHWToLLHDPass()";
  let dependentDialects = ["llhd::LLHDDialect"];
}

//===----------------------------------------------------------------------===//
// StandardToHandshake
//===----------------------------------------------------------------------===//

def HandshakeAnalysis : Pass<"analyze-dataflow", "mlir::ModuleOp"> {
  let summary = "Print resource (operation) statistics";
  let constructor = "circt::createHandshakeAnalysisPass()";
}

def HandshakeDataflow : Pass<"create-dataflow", "mlir::ModuleOp"> {
  let summary = "Convert standard MLIR into dataflow IR";
  let constructor = "circt::createHandshakeDataflowPass()";
  let dependentDialects = ["handshake::HandshakeOpsDialect"];
}

def HandshakeCanonicalize : Pass<"canonicalize-dataflow", "handshake::FuncOp"> {
  let summary = "Canonicalize handshake IR";
  let constructor = "circt::createHandshakeCanonicalizePass()";
}

def HandshakeRemoveBlock : Pass<"remove-block-structure", "handshake::FuncOp"> {
  let summary = "Remove block structure in handshake IR";
  let constructor = "circt::createHandshakeRemoveBlockPass()";
}

def HandshakeInsertBuffer
  : Pass<"handshake-insert-buffer", "handshake::FuncOp"> {
  let summary = "Insert buffers to break graph cycles";
  let constructor = "circt::createHandshakeInsertBufferPass()";
  let options = [
    ListOption<"strategies", "strategies", "std::string",
               "List of strategies to apply. Possible values are: cycles",
               "llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated">
  ];
}

//===----------------------------------------------------------------------===//
// StandardToStaticLogic
//===----------------------------------------------------------------------===//

def CreatePipeline : Pass<"create-pipeline", "mlir::FuncOp"> {
  let summary = "Create StaticLogic pipeline operations";
  let constructor = "circt::createCreatePipelinePass()";
  let dependentDialects = ["staticlogic::StaticLogicDialect"];
}

#endif // CIRCT_CONVERSION_PASSES_TD
