// RUN: circt-opt --lower-sim-to-sv %s | FileCheck %s

// CHECK-LABEL: hw.module @NonProcedural
hw.module @NonProcedural(in %clock: !seq.clock, in %cond: i1) {
  // CHECK-NEXT: sv.ifdef @SYNTHESIS {
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   [[CLOCK:%.+]] = seq.from_clock %clock
  // CHECK-NEXT:   sv.always posedge [[CLOCK]] {
  // CHECK-NEXT:     sv.if %cond {

  // CHECK-NEXT: sv.finish 1
  sim.clocked_exit %clock, %cond, success, verbose
  // CHECK-NEXT: sv.fatal 1
  sim.clocked_exit %clock, %cond, failure, verbose
  // CHECK-NEXT: sv.finish 0
  sim.clocked_exit %clock, %cond, success, quiet
  // CHECK-NEXT: sv.fatal 0
  sim.clocked_exit %clock, %cond, failure, quiet
  // CHECK-NEXT: sv.stop 1
  sim.clocked_pause %clock, %cond, verbose
  // CHECK-NEXT: sv.stop 0
  sim.clocked_pause %clock, %cond, quiet

  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
}

// CHECK-LABEL: hw.module @Procedural
hw.module @Procedural() {
  // CHECK-NEXT: sv.initial {
  sv.initial {
    // CHECK-NEXT: sv.ifdef.procedural @SYNTHESIS {
    // CHECK-NEXT: } else {

    // CHECK-NEXT: sv.finish 1
    sim.exit success, verbose
    // CHECK-NEXT: sv.fatal 1
    sim.exit failure, verbose
    // CHECK-NEXT: sv.finish 0
    sim.exit success, quiet
    // CHECK-NEXT: sv.fatal 0
    sim.exit failure, quiet
    // CHECK-NEXT: sv.stop 1
    sim.pause verbose
    // CHECK-NEXT: sv.stop 0
    sim.pause quiet

    // CHECK-NEXT: }
  }
  // CHECK-NEXT: }
}
