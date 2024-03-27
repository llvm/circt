// RUN: circt-opt --lower-sim-to-sv %s | FileCheck %s

// CHECK-LABEL: hw.module @finish
hw.module @finish(in %clock : !seq.clock, in %cond : i1) {
  // CHECK:      [[CLK_SV:%.+]] = seq.from_clock %clock
  // CHECK-NEXT: sv.ifdef @SYNTHESIS {
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   sv.always posedge [[CLK_SV]] {
  // CHECK-NEXT:     sv.if %cond {
  // CHECK-NEXT:       sv.finish 1
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  sim.finish %clock, %cond
}

// CHECK-LABEL: hw.module @fatal
hw.module @fatal(in %clock : !seq.clock, in %cond : i1) {
  // CHECK:      [[CLK_SV:%.+]] = seq.from_clock %clock
  // CHECK-NEXT: sv.ifdef @SYNTHESIS {
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   sv.always posedge [[CLK_SV]] {
  // CHECK-NEXT:     sv.if %cond {
  // CHECK-NEXT:       sv.fatal 1
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  sim.fatal %clock, %cond
}
