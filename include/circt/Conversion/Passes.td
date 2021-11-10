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
  let description = "This pass lowers SCF / standard to Calyx.";
  let constructor = "circt::createSCFToCalyxPass()";
  let dependentDialects = [
    "calyx::CalyxDialect", "::mlir::scf::SCFDialect", "hw::HWDialect",
    "comb::CombDialect"
  ];
  let options =
      [Option<"topLevelFunctionOpt", "top-level-function", "std::string", "",
              "Identifier of top-level function to be the entry-point component"
              " of the Calyx program.">];
}

//===----------------------------------------------------------------------===//
// CalyxToHW
//===----------------------------------------------------------------------===//

def CalyxToHW : Pass<"lower-calyx-to-hw", "mlir::ModuleOp"> {
  let summary = "Lower Calyx to HW";
  let description = "This pass lowers Calyx to HW.";
  let constructor = "circt::createCalyxToHWPass()";
  let dependentDialects = ["comb::CombDialect", "hw::HWDialect",
                           "sv::SVDialect"];
}

//===----------------------------------------------------------------------===//
// CHIRRTLToFIRRTL
//===----------------------------------------------------------------------===//

def LowerCHIRRTLToFIRRTL : Pass<"lower-chirrtl-to-firrtl", "firrtl::FModuleOp"> {
  let summary = "Infer the memory ports of SeqMem and CombMem";
  let description = [{
      This pass finds the CHIRRTL behavioral memories and their ports, and
      transforms them into standard FIRRTL memory operations.  For each
      `seqmem` or `combmem`, a new memory is created.  For every `memoryport`
      operation using a CHIRRTL memory, a memory port is defined on the
      new standard memory.

      The direction or kind of the port is inferred from how each of the memory
      ports is used in the IR. If a memory port is only written to, it becomes
      a `Write` port.  If a memory port is only read from, it become a `Read`
      port.  If it is used both ways, it becomes a `ReadWrite` port.

      `Write`, `ReadWrite` and combinational `Read` ports are disabled by
      default, but then enabled when the CHIRRTL memory port is declared.
      Sequential `Read` ports have more complicated enable inference:

      1. If a wire or register is used as the index of the memory port, then
         the memory is enabled whenever a non-invalid value is driven to the
         address. 
      2. If a node is used as the index of the memory port, then the memory is
         enabled at the declaration of the node.
      3. In all other cases, the memory is never enabled.

      In the first two cases, they can easily produce a situation where we try
      to enable the memory before it is even declared. This produces a
      compilation error.

  }];
  let constructor = "circt::createLowerCHIRRTLToFIRRTLPass()";
  let dependentDialects = ["firrtl::FIRRTLDialect"];
}

//===----------------------------------------------------------------------===//
// FIRRTLToHW
//===----------------------------------------------------------------------===//

def LowerFIRRTLToHW : Pass<"lower-firrtl-to-hw", "mlir::ModuleOp"> {
  let summary = "Lower FIRRTL to HW";
  let description =
    "Lower a module of FIRRTL dialect to the HW dialect family.";
  let constructor = "circt::createLowerFIRRTLToHWPass()";
  let dependentDialects = ["comb::CombDialect", "hw::HWDialect",
                           "sv::SVDialect"];
  let options = [
    Option<"enableAnnotationWarning", "warn-on-unprocessed-annotations",
           "bool", "false",
    "Emit warnings on unprocessed annotations during lower-to-hw pass">
  ];
}

//===----------------------------------------------------------------------===//
// HandshakeToFIRRTL
//===----------------------------------------------------------------------===//

def HandshakeToFIRRTL : Pass<"lower-handshake-to-firrtl", "mlir::ModuleOp"> {
  let summary = "Lower Handshake to FIRRTL";
  let description = "Lower Handshake to FIRRTL.";
  let constructor = "circt::createHandshakeToFIRRTLPass()";
  let dependentDialects = ["firrtl::FIRRTLDialect"];
}

//===----------------------------------------------------------------------===//
// LLHDToLLVM
//===----------------------------------------------------------------------===//

def ConvertLLHDToLLVM : Pass<"convert-llhd-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert LLHD to LLVM";
  let description = "This pass translates LLHD to LLVM.";
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

def HandshakeAnalysis : Pass<"handshake-analyze", "mlir::ModuleOp"> {
  let summary = "Print resource (operation) statistics";
  let constructor = "circt::createHandshakeAnalysisPass()";
}

def HandshakeDataflow : Pass<"lower-std-to-handshake", "mlir::ModuleOp"> {
  let summary = "Lower Standard MLIR into Handshake IR";
  let constructor = "circt::createHandshakeDataflowPass()";
  let dependentDialects = ["handshake::HandshakeDialect"];
}

def HandshakeCanonicalize : Pass<"handshake-canonicalize", "handshake::FuncOp"> {
  let summary = "Canonicalize Handshake IR";
  let constructor = "circt::createHandshakeCanonicalizePass()";
}

def HandshakeRemoveBlock : Pass<"handshake-remove-block-structure", "handshake::FuncOp"> {
  let summary = "Remove block structure in Handshake IR";
  let constructor = "circt::createHandshakeRemoveBlockPass()";
}

def HandshakeInsertBuffer
  : Pass<"handshake-insert-buffer", "handshake::FuncOp"> {
  let summary = "Insert buffers to break graph cycles";
  let constructor = "circt::createHandshakeInsertBufferPass()";
  let options = [
    ListOption<"strategies", "strategies", "std::string",
               "List of strategies to apply. Possible values are: cycles, all (default)",
               "llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated">,
    Option<"bufferSize", "buffer-size", "unsigned", /*default=*/"2",
           "Number of slots in each buffer">,
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
