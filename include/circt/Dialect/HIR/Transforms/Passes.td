#ifndef CIRCT_DIALECT_HIR_TRANSFORMS_PASSES
#define CIRCT_DIALECT_HIR_TRANSFORMS_PASSES
include "mlir/Pass/PassBase.td"

def SeqScheduler : Pass<"hir-seq-scheduler", "hir::FuncOp"> {
  let summary = "A Simple sequential scheduler for HIR.";
  let description = [{This pass populates the schedule in an unscheduled HIR 
  program assuming that the original program was sequential. The scheduler 
  assumes a simple processor like model - only one operation can happen at 
  a time and all memrefs alias.}];

  let constructor = "circt::hir::createSeqSchedulerPass()";
}

def MemrefLowering : Pass<"hir-memref-lowering", "hir::FuncOp"> {
  let summary = "Lower hir.memref to hir.bus.";
  let description = [{This pass lowers hir.memref to hir.bus for codegen.}];

  let constructor = "circt::hir::createMemrefLoweringPass()";
}

def ScheduleVerifier : Pass<"hir-schedule-verifier", "hir::FuncOp"> {
  let summary = "Verify that the schedule is correct";
  let description = [{This pass finds anomalies in HIR schedules.}];

  let constructor = "circt::hir::createScheduleVerificationPass()";
}

def LoopUnroll : Pass<"hir-loop-unroll", "hir::FuncOp"> {
  let summary = "Unroll loop body";
  let description = [{This pass unrolls the body of the UnrollForOp and removes
  the op.
  }];

  let constructor = "circt::hir::createLoopUnrollPass()";
}
#endif // CIRCT_DIALECT_HIR_TRANSFORMS_PASSES
