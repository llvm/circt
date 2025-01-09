// RUN: circt-opt %s -lower-loopschedule-to-calyx -canonicalize -split-input-file | FileCheck %s

// This will introduce duplicate groups; these should be subsequently removed during canonicalization.

// CHECK:      calyx.while %std_lt_0.out with @bb0_0 {
// CHECK-NEXT:  calyx.par {
// CHECK-NEXT:   calyx.enable @bb0_1
// CHECK-NEXT:  }
// CHECK-NEXT: } {bound = 19 : i64}
module {
  func.func @foo() attributes {} {
    %const = arith.constant 1 : index
    loopschedule.pipeline II =  1 trip_count =  20 iter_args(%counter = %const) : (index) -> () {
      %latch = arith.cmpi ult, %counter, %const : index
      loopschedule.register %latch : i1
    } do {
      %S0 = loopschedule.pipeline.stage start = 0 {
        %op = arith.addi %counter, %const : index
        loopschedule.register %op : index
      } : index
      %S1 = loopschedule.pipeline.stage start = 1 {
        loopschedule.register %S0: index
      } : index
      loopschedule.terminator iter_args(%S0), results() : (index) -> ()
    }
    return
  }
}
