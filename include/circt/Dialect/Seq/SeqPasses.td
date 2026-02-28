//===-- SeqPasses.td - Seq pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for passes that work on the Seq dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SEQ_SEQPASSES
#define CIRCT_DIALECT_SEQ_SEQPASSES

include "mlir/Pass/PassBase.td"

def LowerSeqFIFO : Pass<"lower-seq-fifo", "hw::HWModuleOp"> {
  let summary = "Lower seq.fifo ops";
  let constructor = "circt::seq::createLowerSeqFIFOPass()";
  let dependentDialects = [
    "circt::hw::HWDialect",
    "circt::comb::CombDialect",
    "circt::verif::VerifDialect"
  ];
}

def LowerSeqHLMem: Pass<"lower-seq-hlmem", "hw::HWModuleOp"> {
  let summary = "Lowers seq.hlmem operations.";
  let constructor = "circt::seq::createLowerSeqHLMemPass()";
  let dependentDialects = ["circt::sv::SVDialect"];
}

def ExternalizeClockGate: Pass<"externalize-clock-gate", "mlir::ModuleOp"> {
  let summary = "Convert seq.clock_gate ops into hw.module.extern instances";
  let constructor = "circt::seq::createExternalizeClockGatePass()";
  let dependentDialects = ["circt::hw::HWDialect", "circt::comb::CombDialect"];
  let options = [
    Option<"moduleName", "name", "std::string", "\"CKG\"",
           "Name of the external clock gate module">,
    Option<"inputName", "input", "std::string", "\"I\"",
           "Name of the clock input">,
    Option<"outputName", "output", "std::string", "\"O\"",
           "Name of the gated clock output">,
    Option<"enableName", "enable", "std::string", "\"E\"",
           "Name of the enable input">,
    Option<"testEnableName", "test-enable", "std::string", "\"TE\"",
           "Name of the optional test enable input">,
    Option<"instName", "instance-name", "std::string", "\"ckg\"",
           "Name of the generated instances">
  ];
  let statistics = [
    Statistic<"numClockGatesConverted", "num-clock-gates-converted",
      "Number of clock gates converted to external module instances">
  ];
}

def HWMemSimImpl : Pass<"hw-memory-sim", "ModuleOp"> {
  let summary = "Implement FIRRTMMem memories nodes with simulation model";
  let description = [{
    This pass replaces generated module nodes of type FIRRTLMem with a model
    suitable for simulation.
  }];

  let constructor = "circt::seq::createHWMemSimImplPass()";
  let dependentDialects = [
    "circt::emit::EmitDialect",
    "circt::hw::HWDialect",
    "circt::sv::SVDialect",
  ];

  let options = [
    Option<"disableMemRandomization", "disable-mem-randomization", "bool", "false",
            "Disable emission of memory randomization code">,
    Option<"disableRegRandomization", "disable-reg-randomization", "bool", "false",
            "Disable emission of register randomization code">,
    Option<"replSeqMem", "repl-seq-mem", "bool",
                "false", "Prepare seq mems for macro replacement">,
    Option<"readEnableMode", "read-enable-mode", "::circt::seq::ReadEnableMode",
           "::circt::seq::ReadEnableMode::Undefined",
           "specify the behaviour of the read enable signal", [{
              ::llvm::cl::values(
              clEnumValN(::circt::seq::ReadEnableMode::Ignore, "ignore",
                         "Ignore the read enable signal"),
              clEnumValN(::circt::seq::ReadEnableMode::Zero, "zero",
                         "Output zeros when reads are disabled"),
              clEnumValN(::circt::seq::ReadEnableMode::Undefined, "undefined",
                         "Output X when reads are disbled"))
           }]>,
    Option<"addMuxPragmas", "add-mux-pragmas", "bool", "false",
            "Add mux pragmas to memory reads">,
    Option<"addVivadoRAMAddressConflictSynthesisBugWorkaround",
           "add-vivado-ram-address-conflict-synthesis-bug-workaround", "bool", "false",
            "Add a vivado attribute to specify a ram style of an array register">

   ];
}

def LowerSeqShiftReg : Pass<"lower-seq-shiftreg", "hw::HWModuleOp"> {
  let summary = "Lower seq.shiftreg ops";
  let description = [{
    Default pass for lowering shift register operations. This will lower
    into a chain of `seq.compreg.ce` operations.
    Note that this is _not_ guaranteed to be a performant implementation,
    but instead a default, fallback lowering path which is guaranteed to
    provide a semantically valid path to verilog emissions.
    Users are expected to provide a custom lowering pass to maps `seq.shiftreg`
    operations to target-specific primitives.
  }];
  let constructor = "circt::seq::createLowerSeqShiftRegPass()";
  let dependentDialects = ["circt::hw::HWDialect"];
}

def RegOfVecToMem : Pass<"seq-reg-of-vec-to-mem", "hw::HWModuleOp"> {
  let summary = "Convert register arrays to FIRRTL memories";
  let description = [{
    This pass identifies register arrays that follow memory access patterns
    and converts them to seq.firmem operations. It looks for patterns where:
    1. A register array is updated via array_inject operations
    2. The array is read via array_get operations  
    3. Updates are controlled by enable signals through mux operations
    4. Read and write operations use the same clock
  }];
}

#endif // CIRCT_DIALECT_SEQ_SEQPASSES
