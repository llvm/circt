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
// AffineToStaticLogic
//===----------------------------------------------------------------------===//

def AffineToStaticLogic : Pass<"convert-affine-to-staticlogic", "mlir::func::FuncOp"> {
  let summary = "Convert Affine dialect to StaticLogic pipelines";
  let description = [{
    This pass analyzes Affine loops and control flow, creates a Scheduling
    problem using the Calyx operator library, solves the problem, and lowers
    the loops to a StaticLogic pipeline.
  }];
  let constructor = "circt::createAffineToStaticLogic()";
  let dependentDialects = [
    "circt::staticlogic::StaticLogicDialect",
    "mlir::arith::ArithmeticDialect",
    "mlir::cf::ControlFlowDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "mlir::func::FuncDialect"
  ];
}

//===----------------------------------------------------------------------===//
// ExportVerilog and ExportSplitVerilog
//===----------------------------------------------------------------------===//

def ExportVerilog : Pass<"export-verilog", "mlir::ModuleOp"> {
  let summary = "Emit the IR to a (System)Verilog file";
  let description = [{
      This pass creates empty module bodies for external modules.  This is
      useful for linting to eliminate missing file errors.
  }];

  let constructor = "createExportVerilogPass()";
  let dependentDialects = [
    "circt::sv::SVDialect", "circt::comb::CombDialect", "circt::hw::HWDialect"
  ];
}

def ExportSplitVerilog : Pass<"export-split-verilog", "mlir::ModuleOp"> {
  let summary = "Emit the IR to a (System)Verilog directory of files";
  let description = [{
    This pass generates (System)Verilog for the current design, mutating it
    where necessary to be valid Verilog.
  }];

  let constructor = "createExportSplitVerilogPass()";
  let dependentDialects = [
    "circt::sv::SVDialect", "circt::comb::CombDialect", "circt::hw::HWDialect"
  ];

  let options = [
    Option<"directoryName", "dir-name", "std::string",
            "", "Directory to emit into">
   ];
}

//===----------------------------------------------------------------------===//
// SCFToCalyx
//===----------------------------------------------------------------------===//

def SCFToCalyx : Pass<"lower-scf-to-calyx", "mlir::ModuleOp"> {
  let summary = "Lower SCF/Standard to Calyx";
  let description = [{
    This pass lowers SCF / standard to Calyx.
  }];
  let constructor = "circt::createSCFToCalyxPass()";
  let dependentDialects = [
    "calyx::CalyxDialect", "::mlir::scf::SCFDialect", "hw::HWDialect",
    "comb::CombDialect"
  ];
  let options = [
    Option<"topLevelFunctionOpt", "top-level-function", "std::string", "",
            "Identifier of top-level function to be the entry-point component"
            " of the Calyx program.">,
    Option<"ciderSourceLocationMetadata", "cider-source-location-metadata", "bool", "",
            "Whether to track source location for the Cider debugger.">
  ];
}

//===----------------------------------------------------------------------===//
// StaticLogicToCalyx
//===----------------------------------------------------------------------===//

def StaticLogicToCalyx : Pass<"lower-static-logic-to-calyx", "mlir::ModuleOp"> {
  let summary = "Lower StaticLogic to Calyx";
  let description = [{
    This pass lowers StaticLogic to Calyx.
  }];
  let constructor = "circt::createStaticLogicToCalyxPass()";
  let dependentDialects = [
    "calyx::CalyxDialect", "::mlir::scf::SCFDialect", "hw::HWDialect",
    "comb::CombDialect"
  ];
  let options = [
    Option<"topLevelFunctionOpt", "top-level-function", "std::string", "",
            "Identifier of top-level function to be the entry-point component"
            " of the Calyx program.">,
    Option<"ciderSourceLocationMetadata", "cider-source-location-metadata", "bool", "",
            "Whether to track source location for the Cider debugger.">
  ];
}

//===----------------------------------------------------------------------===//
// CalyxToHW
//===----------------------------------------------------------------------===//

def CalyxToHW : Pass<"lower-calyx-to-hw", "mlir::ModuleOp"> {
  let summary = "Lower Calyx to HW";
  let description = [{
    This pass lowers Calyx to HW.
  }];
  let constructor = "circt::createCalyxToHWPass()";
  let dependentDialects = ["comb::CombDialect", "hw::HWDialect",
                           "seq::SeqDialect", "sv::SVDialect"];
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
  let options = [
    Option<"enableAnnotationWarning", "warn-on-unprocessed-annotations",
           "bool", "false",
    "Emit warnings on unprocessed annotations during lower-to-hw pass">,
    Option<"emitChiselAssertsAsSVA", "emit-chisel-asserts-as-sva",
           "bool", "false","Convert all Chisel asserts to SVA">
  ];
}

//===----------------------------------------------------------------------===//
// HandshakeToHW
//===----------------------------------------------------------------------===//

def HandshakeToHW : Pass<"lower-handshake-to-hw", "mlir::ModuleOp"> {
  let summary = "Lower Handshake to ESI/HW/Comb/Seq";
  let description = [{
    Lower Handshake to ESI/HW/Comb/Seq.
  }];
  let constructor = "circt::createHandshakeToHWPass()";
  let dependentDialects = ["hw::HWDialect", "esi::ESIDialect"];
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
  let options = [
    Option<"enableFlattening", "flatten", "bool", "false",
    "Flattens the generated FIRRTL component by inlining all dataflow component"
    " instantiations into the top module.">,
  ];
}

//===----------------------------------------------------------------------===//
// Moore to Comb/HW/LLHD
//===----------------------------------------------------------------------===//

def ConvertMooreToCore : Pass<"convert-moore-to-core", "mlir::ModuleOp"> {
  let summary = "Convert Moore to Core";
  let description = [{
    This pass translates Moore to the core dialects (Comb/HW/LLHD).
  }];
  let constructor = "circt::createConvertMooreToCorePass()";
  let dependentDialects = ["comb::CombDialect", "hw::HWDialect",
                           "llhd::LLHDDialect"];
}

//===----------------------------------------------------------------------===//
// LLHDToLLVM
//===----------------------------------------------------------------------===//

def ConvertLLHDToLLVM : Pass<"convert-llhd-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert LLHD to LLVM";
  let description = [{
    This pass translates LLHD to LLVM."
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

def StandardToHandshake : Pass<"lower-std-to-handshake", "mlir::ModuleOp"> {
  let summary = "Lower Standard MLIR into Handshake IR";
  let constructor = "circt::createStandardToHandshakePass()";
  let dependentDialects = ["handshake::HandshakeDialect"];
  let options =
    [Option<"sourceConstants", "source-constants", "bool", "false",
            "If true, will connect constants to source operations instead of "
            "to the control network. May reduce the size of the final circuit.">,
     Option<"disableTaskPipelining", "disable-task-pipelining", "bool", "false",
            "If true, will disable support for task pipelining. This relaxes the "
            "restrictions put on the structure of the input CDFG. Disabling "
            "task pipelining may severely reduce kernel II.">];
}

def HandshakeRemoveBlock : Pass<"handshake-remove-block-structure", "handshake::FuncOp"> {
  let summary = "Remove block structure in Handshake IR";
  let constructor = "circt::createHandshakeRemoveBlockPass()";
}

//===----------------------------------------------------------------------===//
// StandardToStaticLogic
//===----------------------------------------------------------------------===//

def CreatePipeline : Pass<"create-pipeline", "mlir::func::FuncOp"> {
  let summary = "Create StaticLogic pipeline operations";
  let constructor = "circt::createCreatePipelinePass()";
  let dependentDialects = ["staticlogic::StaticLogicDialect"];
}

#endif // CIRCT_CONVERSION_PASSES_TD
