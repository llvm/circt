// RUN: circt-opt -convert-affine-to-staticlogic %s | FileCheck %s

// CHECK-LABEL: func @minimal
func @minimal() {
  // Setup constants.
  // CHECK: %[[LB:.+]] = arith.constant 0 : [[ITER_TYPE:.+]]
  // CHECK: %[[UB:.+]] = arith.constant 10 : [[ITER_TYPE]]
  // CHECK: %[[STEP:.+]] = arith.constant 1 : [[ITER_TYPE]]

  // Pipeline header.
  // CHECK: staticlogic.pipeline.while #staticlogic.II<1> iter_args(%[[ITER_ARG:.+]] = %[[LB]]) : ([[ITER_TYPE]]) -> ()

  // Condition block.
  // CHECK: %[[COND_RESULT:.+]] = arith.cmpi ult, %[[ITER_ARG]]
  // CHECK: staticlogic.pipeline.register %[[COND_RESULT]]

  // First stage.
  // CHECK: %[[STAGE0:.+]] = staticlogic.pipeline.stage
  // CHECK: %[[ITER_INC:.+]] = arith.addi %[[ITER_ARG]], %[[STEP]]
  // CHECK: staticlogic.pipeline.register %[[ITER_INC]]

  // Pipeline terminator.
  // CHECK: staticlogic.pipeline.terminator iter_args(%[[STAGE0]]), results()

  affine.for %arg1 = 0 to 10 {
  }

  return
}
