#ifndef CIRCT_DIALECT_HIR_TRANSFORMS_PASSES
#define CIRCT_DIALECT_HIR_TRANSFORMS_PASSES
include "mlir/Pass/PassBase.td"

def IndexLowering : Pass<"hir-lower-index", "hir::FuncOp"> {
  let summary = "Converts the IndexType to an IntegerType of concrete size.";
  let description = [{IndexType of for loops (except unrolled for loops) is converted to a concrete type.}];

  let constructor = "circt::hir::createIndexLoweringPass()";
}

def RegisterAllocation : Pass<"hir-register-alloc", "hir::FuncOp"> {
  let summary = "Register allocation pass for HIR programs.";
  let description = [{Allocates registers for pipeline balancing using hir.delay and hir.latch instructions.}];

  let constructor = "circt::hir::createRegisterAllocationPass()";
}

def SeqScheduler : Pass<"hir-seq-scheduler", "hir::FuncOp"> {
  let summary = "A Simple sequential scheduler for HIR.";
  let description = [{This pass populates the schedule in an unscheduled HIR 
  program assuming that the original program was sequential. The scheduler 
  assumes a simple processor like model - only one operation can happen at 
  a time and all memrefs alias.}];

  let constructor = "circt::hir::createSeqSchedulerPass()";
}

def MemrefLowering : Pass<"hir-lower-memref", "hir::FuncOp"> {
  let summary = "Lower hir.memref to hir.bus.";
  let description = [{This pass lowers hir.memref to hir.bus for codegen.}];

  let constructor = "circt::hir::createMemrefLoweringPass()";
  let dependentDialects = ["hir::HIRDialect"];
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
