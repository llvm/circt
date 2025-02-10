// RUN: circt-opt --canonicalize %s | FileCheck %s

// CHECK-LABEL: @HasBeenReset
hw.module @HasBeenReset(in %clock: i1, in %reset: i1) {
  // CHECK-NEXT: %false = hw.constant false
  // CHECK-NEXT: %true = hw.constant true
  %false = hw.constant false
  %true = hw.constant true

  // CHECK-NEXT: %constResetA0 = hw.wire %false
  // CHECK-NEXT: %constResetA1 = hw.wire %false
  // CHECK-NEXT: %constResetS0 = hw.wire %false
  // CHECK-NEXT: %constResetS1 = hw.wire %false
  %r0 = verif.has_been_reset %clock, async %false
  %r1 = verif.has_been_reset %clock, async %true
  %r2 = verif.has_been_reset %clock, sync %false
  %r3 = verif.has_been_reset %clock, sync %true
  %constResetA0 = hw.wire %r0 sym @constResetA0 : i1
  %constResetA1 = hw.wire %r1 sym @constResetA1 : i1
  %constResetS0 = hw.wire %r2 sym @constResetS0 : i1
  %constResetS1 = hw.wire %r3 sym @constResetS1 : i1

  // CHECK-NEXT: [[TMP1:%.+]] = verif.has_been_reset %false, async %reset
  // CHECK-NEXT: [[TMP2:%.+]] = verif.has_been_reset %true, async %reset
  // CHECK-NEXT: %constClockA0 = hw.wire [[TMP1]]
  // CHECK-NEXT: %constClockA1 = hw.wire [[TMP2]]
  // CHECK-NEXT: %constClockS0 = hw.wire %false
  // CHECK-NEXT: %constClockS1 = hw.wire %false
  %c0 = verif.has_been_reset %false, async %reset
  %c1 = verif.has_been_reset %true, async %reset
  %c2 = verif.has_been_reset %false, sync %reset
  %c3 = verif.has_been_reset %true, sync %reset
  %constClockA0 = hw.wire %c0 sym @constClockA0 : i1
  %constClockA1 = hw.wire %c1 sym @constClockA1 : i1
  %constClockS0 = hw.wire %c2 sym @constClockS0 : i1
  %constClockS1 = hw.wire %c3 sym @constClockS1 : i1
}

// CHECK-LABEL: @clockedAssert
hw.module @clockedAssert(in %clock : i1, in %a : i1, in %en : i1) {
  // CHECK: verif.clocked_assert %a if %en, posedge %clock : i1
  %clk = ltl.clock %a, posedge %clock : i1
  verif.assert %clk if %en : !ltl.sequence
}

// CHECK-LABEL: @clockedAssume
hw.module @clockedAssume(in %clock : i1, in %a : i1, in %en : i1) {
  // CHECK: verif.clocked_assume %a if %en, posedge %clock : i1
  %clk = ltl.clock %a, posedge %clock : i1
  verif.assume %clk if %en : !ltl.sequence
}

// CHECK-LABEL: @clockedCover
hw.module @clockedCover(in %clock : i1, in %a : i1, in %en : i1) {
  // CHECK: verif.clocked_cover %a if %en,  posedge %clock : i1
  %clk = ltl.clock %a, posedge %clock : i1
  verif.cover %clk if %en : !ltl.sequence
}

// CHECK-LABEL: @RemoveUnusedSymbolicValues
hw.module @RemoveUnusedSymbolicValues() {
  // CHECK-NOT: verif.symbolic_value
  // CHECK: hw.output
  %0 = verif.symbolic_value : i32
}
