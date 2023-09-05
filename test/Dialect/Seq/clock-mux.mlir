// RUN: circt-opt --lower-seq-to-sv %s | FileCheck %s

// CHECK-LABEL: hw.module @clock_mux
hw.module @clock_mux(%cond : i1, %trueClock: !seq.clock, %falseClock: !seq.clock) -> (out: !seq.clock) {
  // CHECK: [[MUX:%.+]] = comb.mux bin %cond, %trueClock, %falseClock
  // CHECK: hw.output [[MUX]] : i1
  %0 = seq.clock_mux %cond, %trueClock, %falseClock
  hw.output %0 : !seq.clock
}
