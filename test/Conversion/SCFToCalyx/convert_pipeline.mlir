// RUN: circt-opt %s -lower-scf-to-calyx | FileCheck %s

func @minimal() {
  %c0_i64 = arith.constant 0 : i64
  %c10_i64 = arith.constant 10 : i64
  %c1_i64 = arith.constant 1 : i64
  staticlogic.pipeline.while II =  1 iter_args(%arg0 = %c0_i64) : (i64) -> () {
    %0 = arith.cmpi ult, %arg0, %c10_i64 : i64
    staticlogic.pipeline.register %0 : i1
  } do {
    %0 = staticlogic.pipeline.stage  {
      %1 = arith.addi %arg0, %c1_i64 : i64
      staticlogic.pipeline.register %1 : i64
    } : i64
    staticlogic.pipeline.terminator iter_args(%0), results() : (i64) -> ()
  }
  return
}
