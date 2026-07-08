// RUN: circt-opt %s --arc-lower-state | FileCheck %s

// A module-level probe of a constant-offset bit-slice
// (`llhd.prb (llhd.sig.extract %sig from K)`) lowers as a read of the parent
// signal's storage plus an extract of the slice -- the probe-side mirror of
// the slice-drive splice.

// CHECK-LABEL: arc.model @top
hw.module @top(out o : i1) {
  %c0_i32 = hw.constant 0 : i32
  %c3_i5 = hw.constant 3 : i5
  %sig = llhd.sig %c0_i32 : i32
  %slice = llhd.sig.extract %sig from %c3_i5 : <i32> -> <i1>
  // CHECK: [[CUR:%.+]] = arc.state_read %{{.+}} : <i32>
  // CHECK: [[BIT:%.+]] = comb.extract [[CUR]] from 3 : (i32) -> i1
  %bit = llhd.prb %slice : i1
  hw.output %bit : i1
}
