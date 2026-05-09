// RUN: circt-opt --lower-seq-compreg-ce %s --verify-diagnostics | FileCheck %s


// CHECK-LABEL: hw.module @cetest
// CHECK-NEXT:   [[TMP0:%.+]] = hw.constant 0 : i32
// CHECK-NEXT:   [[TMP1:%.+]] = seq.initial() {
// CHECK-NEXT:     [[C0:%.+]] = hw.constant 0 : i32
// CHECK-NEXT:     seq.yield [[C0]] : i32
// CHECK-NEXT:   } : () -> !seq.immutable<i32>
hw.module @cetest(in %clk: !seq.clock, in %rst: i1, in %ce: i1, in %i: i32) {
  %rv = hw.constant 0 : i32
  %init = seq.initial() {
    %c0_i32 = hw.constant 0 : i32
    seq.yield %c0_i32 : i32
  } : () -> !seq.immutable<i32>

  // CHECK-NEXT: [[TMP2:%.+]] = comb.mux %ce, %i, %r0 : i32
  // CHECK-NEXT: %r0 = seq.compreg [[TMP2]], %clk reset %rst, [[TMP0]] : i32
  %r0 = seq.compreg.ce %i, %clk, %ce reset %rst, %rv : i32
  // CHECK-NEXT: [[TMP3:%.+]] = comb.mux %ce, %i, %r1 : i32
  // CHECK-NEXT: %r1 = seq.compreg sym @withinitial [[TMP3]], %clk reset %rst, [[TMP0]] initial [[TMP1]] : i32
  %r1 = seq.compreg.ce sym @withinitial %i, %clk, %ce reset %rst, %rv initial %init : i32

// CHECK-NEXT }
}
