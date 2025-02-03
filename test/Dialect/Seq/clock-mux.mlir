// RUN: circt-opt --lower-seq-to-sv %s | FileCheck %s

// CHECK-LABEL: hw.module @clock_mux
hw.module @clock_mux(in %cond : i1, in %trueClock: !seq.clock, in %falseClock: !seq.clock, out out: !seq.clock) {
  // CHECK: [[MUX:%.+]] = comb.mux %cond, %trueClock, %falseClock
  // CHECK: hw.output [[MUX]] : i1
  %0 = seq.clock_mux %cond, %trueClock, %falseClock
  hw.output %0 : !seq.clock
}
