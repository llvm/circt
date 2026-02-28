// RUN: circt-opt --lower-sim-to-sv %s | FileCheck %s

// CHECK-LABEL: hw.module @NonProcedural
hw.module @NonProcedural(in %clock: !seq.clock, in %cond: i1) {
  // CHECK-NEXT: sv.ifdef @SYNTHESIS {
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   [[CLOCK:%.+]] = seq.from_clock %clock
  // CHECK-NEXT:   sv.always posedge [[CLOCK]] {
  // CHECK-NEXT:     sv.if %cond {

  // CHECK-NEXT: sv.finish 1
  sim.clocked_terminate %clock, %cond, success, verbose
  // CHECK-NEXT: sv.fatal.procedural 1
  sim.clocked_terminate %clock, %cond, failure, verbose
  // CHECK-NEXT: sv.finish 0
  sim.clocked_terminate %clock, %cond, success, quiet
  // CHECK-NEXT: sv.fatal.procedural 0
  sim.clocked_terminate %clock, %cond, failure, quiet
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
    sim.terminate success, verbose
    // CHECK-NEXT: sv.fatal.procedural 1
    sim.terminate failure, verbose
    // CHECK-NEXT: sv.finish 0
    sim.terminate success, quiet
    // CHECK-NEXT: sv.fatal.procedural 0
    sim.terminate failure, quiet
    // CHECK-NEXT: sv.stop 1
    sim.pause verbose
    // CHECK-NEXT: sv.stop 0
    sim.pause quiet

    // CHECK-NEXT: }
  }
  // CHECK-NEXT: }
}

sv.macro.decl @SCHMINTHESIS

// CHECK-LABEL: hw.module @DontMergeIntoIfdefWithDifferentMacro
hw.module @DontMergeIntoIfdefWithDifferentMacro(in %clock: !seq.clock, in %cond: i1) {
  // CHECK: sv.ifdef @SCHMINTHESIS
  sv.ifdef @SCHMINTHESIS {} else {}
  // CHECK:      sv.ifdef @SYNTHESIS {
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   [[CLOCK:%.+]] = seq.from_clock %clock
  // CHECK-NEXT:   sv.always posedge [[CLOCK]] {
  // CHECK-NEXT:     sv.if %cond {
  // CHECK-NEXT:       sv.stop 0
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  sim.clocked_pause %clock, %cond, quiet

  // CHECK: sv.initial
  sv.initial {
    // CHECK: sv.ifdef.procedural @SCHMINTHESIS
    sv.ifdef.procedural @SCHMINTHESIS {} else {}
    // CHECK:      sv.ifdef.procedural @SYNTHESIS {
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   sv.stop 0
    // CHECK-NEXT: }
    sim.pause quiet
  }
}

// CHECK-LABEL: hw.module @DontMergeDifferentClocksOrConditions
hw.module @DontMergeDifferentClocksOrConditions(
  in %clockA: !seq.clock,
  in %clockB: !seq.clock,
  in %condA: i1,
  in %condB: i1
) {
  // CHECK:      sv.ifdef @SYNTHESIS {
  // CHECK-NEXT: } else {

  // CHECK-NEXT: [[TMP:%.+]] = seq.from_clock %clockA
  // CHECK-NEXT: sv.always posedge [[TMP]] {
  // CHECK-NEXT:   sv.if %condA {
  // CHECK-NEXT:     sv.stop 0
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  // CHECK-NEXT: [[TMP:%.+]] = seq.from_clock %clockB
  // CHECK-NEXT: sv.always posedge [[TMP]] {
  // CHECK-NEXT:   sv.if %condA {
  // CHECK-NEXT:     sv.stop 1
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  sim.clocked_pause %clockA, %condA, quiet
  sim.clocked_pause %clockB, %condA, verbose

  // CHECK-NEXT: [[TMP:%.+]] = seq.from_clock %clockA
  // CHECK-NEXT: sv.always posedge [[TMP]] {
  // CHECK-NEXT:   sv.if %condA {
  // CHECK-NEXT:     sv.stop 0
  // CHECK-NEXT:   }
  // CHECK-NEXT:   sv.if %condB {
  // CHECK-NEXT:     sv.stop 1
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  sim.clocked_pause %clockA, %condA, quiet
  sim.clocked_pause %clockA, %condB, verbose
}
