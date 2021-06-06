#ifndef CIRCT_DIALECT_HIR_TRANSFORMS_PASSES
#define CIRCT_DIALECT_HIR_TRANSFORMS_PASSES
include "mlir/Pass/PassBase.td"
def MemrefLowering : Pass<"hir-memref-lowering", "hir::FuncOp"> {
  let summary = "Move time offsets into separate 'delay' op";
  let description = [{This pass lowers memrefs into bus types and function
  calls.  }];

  let constructor = "mlir::hir::createMemrefLoweringPass()";
}

def ScheduleVerifier : Pass<"hir-schedule-verifier", "hir::FuncOp"> {
  let summary = "Verify that the schedule is correct";
  let description = [{This pass finds anomalies in HIR schedules.}];

  let constructor = "mlir::hir::createScheduleVerificationPass()";
}

def LoopUnroll : Pass<"hir-loop-unroll", "hir::FuncOp"> {
  let summary = "Unroll loop body";
  let description = [{This pass unrolls the body of the UnrollForOp and removes
  the op.
  }];

  let constructor = "mlir::hir::createLoopUnrollPass()";
}
#endif // CIRCT_DIALECT_HIR_TRANSFORMS_PASSES
