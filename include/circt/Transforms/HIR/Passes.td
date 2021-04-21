#ifndef CIRCT_DIALECT_HIR_TRANSFORMS_PASSES
#define CIRCT_DIALECT_HIR_TRANSFORMS_PASSES
include "mlir/Pass/PassBase.td"
def Canonicalization : Pass<"hir-canonicalization", "hir::FuncOp"> {
  let summary = "Move time offsets into separate 'delay' op";
  let description = [{This pass removes the time offset parameter into a
  separate hir.delay op to ensure all other ops have just one time parameter.
  }];

  let constructor = "mlir::hir::createCanonicalizationPass()";
}

def ScheduleVerifier : Pass<"hir-schedule-verifier", "hir::FuncOp"> {
  let summary = "Verify that the schedule is correct";
  let description = [{This pass finds anomalies in HIR schedules.}];

  let constructor = "mlir::hir::createScheduleVerificationPass()";
}
#endif // CIRCT_DIALECT_HIR_TRANSFORMS_PASSES
