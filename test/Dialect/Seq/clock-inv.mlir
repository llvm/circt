// RUN: circt-opt --lower-seq-to-sv %s | FileCheck %s

// CHECK-LABEL: @clock_inv
hw.module @clock_inv(in %clk_in : !seq.clock, out clk_out : !seq.clock) {
  // CHECK: [[INV:%.+]] = comb.xor %clk_in, %true : i1
  // CHECK: hw.output [[INV]] : i1
  %0 = seq.clock_inv %clk_in
  hw.output %0 : !seq.clock
}

