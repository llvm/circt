//===-- Passes.td - LLHD pass definition file --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for passes that work on the LLHD dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_LLHD_TRANSFORMS_PASSES
#define CIRCT_DIALECT_LLHD_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def ProcessLowering : Pass<"llhd-process-lowering", "ModuleOp"> {
  let summary = "Lowers LLHD Processes to Entities.";
  let description = [{
    TODO
  }];
}

def MemoryToBlockArgument : Pass<"llhd-memory-to-block-argument",
                                 "hw::HWModuleOp"> {

  let summary = "Promote memory to block arguments.";
  let description = [{
    Promotes memory locations allocated with `llhd.var` to block arguments. This
    enables other optimizations and is required to be able to lower behavioral
    LLHD to structural LLHD. This is because there are no memory model and
    control flow in structural LLHD. After executing this pass, the
    "-llhd-block-argument-to-mux" pass can be used to convert the block
    arguments to multiplexers to enable more control-flow elimination.

    Example:

    ```mlir
    llhd.process {
      %c5 = hw.constant 5 : i32
      %cond = llhd.prb %condsig : !hw.inout<i1>
      %ptr = llhd.var %c5 : i32
      cond_br %cond, ^bb1, ^bb2
    ^bb1:
      %c6 = hw.constant 6 : i32
      llhd.store %ptr, %c6 : !llhd.ptr<i32>
      br ^bb2
    ^bb2:
      %ld = llhd.load %ptr : !llhd.ptr<i32>
      %c-1_i32 = hw.constant -1 : i32
      %res = comb.xor %ld, %c-1_i32 : i32
      llhd.halt
    }
    ```

    is transformed to

    ```mlir
    llhd.process {
      %c5 = hw.constant 5 : i32
      %cond = llhd.prb %condsig : !hw.inout<i1>
      cond_br %cond, ^bb1, ^bb2(%c5 : i32)
    ^bb1:
      %c6 = hw.constant 6 : i32
      br ^bb2(%c6 : i32)
    ^bb2(%arg : i32):
      %c-1_i32 = hw.constant -1 : i32
      %res = comb.xor %arg, %c-1_i32 : i32
      llhd.halt
    }
    ```

  }];
}

def InlineCallsPass : Pass<"llhd-inline-calls", "ModuleOp"> {
  let summary = "Inline all function calls in HW modules";
  let description = [{
    Inlines all `func.call` operations nested within `llhd.combinational`,
    `llhd.process`, and `llhd.final` ops within `hw.module`s. The `func.func`
    definitions are left untouched. After the inlining, MLIR's symbol DCE pass
    may be used to eliminate the function definitions where possible.

    This pass expects the `llhd-wrap-procedural-ops` pass to have already run.
    Otherwise call ops immediately in the module body, a graph region, cannot be
    inlined at all since the graph region cannot accommodate the function's
    control flow.
  }];
  let statistics = [
    Statistic<
      "numInlined", "calls-inlined", "Number of call ops that were inlined"
    >,
  ];
}

def EarlyCodeMotion : Pass<"llhd-early-code-motion", "hw::HWModuleOp"> {
  let summary = "Move side-effect-free instructions and llhd.prb up in the CFG";
  let description = [{
    Moves side-effect-free instructions as far up in the CFG as possible. That
    means to the earliest block where all operands are defined. Special care has
    to be given to the `llhd.prb` instruction (which is the only side-effect
    instruction moved by this pass) as it must stay in the same temporal region,
    because otherwise it might sample an older or newer state of the signal.
    This pass is designed as a preparatory pass for the Temporal Code Motion
    pass to be able to move the `llhd.drv` operations in a single TR exiting
    block without having to move operations defining the operands used by the
    `llhd.drv`. It also enables total control flow elimination as the `llhd.prb`
    instructions would not be moved by other canonicalization passes.
  }];
}

def Sig2Reg : Pass<"llhd-sig2reg", "hw::HWModuleOp"> {
  let summary = "promote LLHD signals to SSA values";

  let dependentDialects = ["circt::hw::HWDialect", "circt::comb::CombDialect"];
}

def Mem2RegPass : Pass<"llhd-mem2reg"> {
  let summary = "Promotes memory and signal slots into values.";
  let dependentDialects = [
    "comb::CombDialect",
    "hw::HWDialect",
    "mlir::cf::ControlFlowDialect",
  ];
}

def TemporalCodeMotion : Pass<"llhd-temporal-code-motion", "hw::HWModuleOp"> {
  let summary = "move drive operations to the exit basic block in processes";
  let description = [{
    This pass uses the temporal region analysis to transform the IR such that
    every temporal region has a unique exit block and moves all 'llhd.drv'
    operations in a temporal region into its exit block by adjusting the
    enable operand.
    Furthermore, it combines 'llhd.drv' operations driving the same signal with
    the same delay by multiplexing the driven value according to their enable
    operands.
    This pass assumes that the early code motion pass has been run beforehand.
    Otherwise, dominance errors are to be expected.
  }];
}

def Desequentialization : Pass<"llhd-desequentialize", "hw::HWModuleOp"> {
  let summary = "convert sequential processes to registers";
  let description = [{
    Analyzes the drive conditions of all drives in sequential processes and
    tries to convert them to registers.
    This is not always possible because a process might consist of more than two
    temporal regions, or the drive condition might depend on values sampled in the
    past that are not a clock, or the condition depends on the conjunction of two
    clocks which is not supported by regular registers, etc.
    If all drives in a process were successfully lowered to registers, the process
    is inlined into the surrounding module.
  }];

  let options = [
    Option<"maxPrimitives", "max-primitives", "unsigned", "8",
           "The maximum number of primitives to analyze.">,
  ];

  let dependentDialects = [
    "comb::CombDialect", "hw::HWDialect", "seq::SeqDialect"
  ];
}

def HoistSignalsPass : Pass<"llhd-hoist-signals"> {
  let summary = "Hoist probes and promote drives to process results";
}

def LowerProcessesPass : Pass<"llhd-lower-processes", "hw::HWModuleOp"> {
  let summary = "Convert process ops to combinational ops where possible";
}

def DeseqPass : Pass<"llhd-deseq", "hw::HWModuleOp"> {
  let summary = "Convert sequential processes to registers";
  let dependentDialects = [
    "comb::CombDialect",
    "hw::HWDialect",
    "seq::SeqDialect",
  ];
}

def CombineDrivesPass : Pass<"llhd-combine-drives", "hw::HWModuleOp"> {
  let summary = "Combine scalar drives into aggregate drives";
  let description = [{
    If individual drives cover all of an aggregate signal's fields, merge them
    into a single drive of the whole aggregate value.
  }];
  let dependentDialects = [
    "comb::CombDialect",
    "hw::HWDialect",
  ];
}

def WrapProceduralOpsPass : Pass<"llhd-wrap-procedural-ops", "hw::HWModuleOp"> {
  let summary = "Wrap procedural ops in modules to make them inlinable";
  let description = [{
    Operations such as `func.call` or `scf.if` may appear in an `hw.module` body
    directly. They cannot be inlined into the module though, since the inlined
    function or result of converting the SCF ops to the CF dialect may create
    control-flow operations with multiple blocks. This pass wraps such
    operations in `llhd.combinational` ops to give them an SSACFG region to
    inline into.
  }];
  let dependentDialects = [
    "llhd::LLHDDialect",
  ];
  let statistics = [
    Statistic<
      "numOpsWrapped", "ops-wrapped", "Number of procedural ops wrapped"
    >,
  ];
}

#endif // CIRCT_DIALECT_LLHD_TRANSFORMS_PASSES
