// RUN: circt-opt %s -lower-loopschedule-to-calyx -canonicalize -split-input-file | FileCheck %s

// This will introduce duplicate groups; these should be subsequently removed.

// CHECK:      calyx.while %std_lt_0.out with @bb0_0 {
// CHECK-NEXT:  calyx.par {
// CHECK-NEXT:   calyx.enable @bb0_1
// CHECK-NEXT:  }
// CHECK-NEXT: }
module {
  func.func @foo() attributes {} {
    %const = arith.constant 1 : index
    loopschedule.pipeline II = 1 trip_count = 20 iter_args(%counter = %const) : (index) -> () {
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

// -----

// Stage pipeline registers passed directly to the next stage 
// should also be updated when used in computations.

// CHECK:      calyx.group @bb0_2 {
// CHECK-NEXT:   calyx.assign %std_add_1.left = %while_0_arg0_reg.out : i32
// CHECK-NEXT:   calyx.assign %std_add_1.right = %c1_i32 : i32
// CHECK-NEXT:   calyx.assign %stage_1_register_0_reg.in = %std_add_1.out : i32
// CHECK-NEXT:   calyx.assign %stage_1_register_0_reg.write_en = %true : i1
// CHECK-NEXT:   calyx.group_done %stage_1_register_0_reg.done : i1
// CHECK-NEXT: }
module {
  func.func @foo() attributes {} {
    %const = arith.constant 1 : index
    loopschedule.pipeline II = 1 trip_count = 20 iter_args(%counter = %const) : (index) -> () {
      %latch = arith.cmpi ult, %counter, %const : index
      loopschedule.register %latch : i1
    } do {
      %S0 = loopschedule.pipeline.stage start = 0 {
        %op = arith.addi %counter, %const : index
        loopschedule.register %op : index
      } : index
      %S1 = loopschedule.pipeline.stage start = 1 {
        %math = arith.addi %S0, %const : index
        loopschedule.register %math : index
      } : index
      loopschedule.terminator iter_args(%S0), results() : (index) -> ()
    }
    return
  }
}

