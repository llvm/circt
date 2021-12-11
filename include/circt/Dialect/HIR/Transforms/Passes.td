#ifndef CIRCT_DIALECT_HIR_TRANSFORMS_PASSES
#define CIRCT_DIALECT_HIR_TRANSFORMS_PASSES
include "mlir/Pass/PassBase.td"

def SimplifyCtrl : Pass<"hir-simplify-ctrl", "hir::FuncOp"> {
  let summary = "Simplifies ForOp and IfOp.";

  let constructor = "circt::hir::createSimplifyCtrlPass()";
  let dependentDialects = ["circt::hw::HWDialect","circt::comb::CombDialect"];
}

def MemrefLowering : Pass<"hir-lower-memref", "mlir::ModuleOp"> {
  let summary = "Lower hir.memref to hir.bus.";
  let description = [{This pass lowers hir.memref to hir.bus for codegen.}];

  let constructor = "circt::hir::createMemrefLoweringPass()";
  let dependentDialects = ["mlir::arith::ArithmeticDialect","circt::hw::HWDialect","circt::comb::CombDialect","hir::HIRDialect"];
}

def VerifySchedule : Pass<"hir-verify-schedule", "hir::FuncOp"> {
  let summary = "Verify that the schedule is correct";
  let description = [{This pass finds anomalies in HIR schedules.}];

  let constructor = "circt::hir::createVerifySchedulePass()";
}

def LoopUnroll : Pass<"hir-loop-unroll", "hir::FuncOp"> {
  let summary = "Unroll loop body";
  let description = [{This pass unrolls the body of the UnrollForOp and removes
  the op.
  }];

  let constructor = "circt::hir::createLoopUnrollPass()";
}
#endif // CIRCT_DIALECT_HIR_TRANSFORMS_PASSES
