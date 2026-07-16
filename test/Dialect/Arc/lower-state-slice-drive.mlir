// RUN: circt-opt %s --arc-lower-state | FileCheck %s

// A module-level drive of a constant-offset bit-slice
// (`llhd.drv (llhd.sig.extract %sig from K)`) lowers as a read-modify-write
// splice into the parent signal's storage.

// CHECK-LABEL: arc.model @top
hw.module @top(in %v : i1) {
  %c0_i32 = hw.constant 0 : i32
  %c1_i5 = hw.constant 1 : i5
  %t = llhd.constant_time <0ns, 0d, 1e>
  %sig = llhd.sig %c0_i32 : i32
  %slice = llhd.sig.extract %sig from %c1_i5 : <i32> -> <i1>
  // CHECK: [[CUR:%.+]] = arc.state_read %{{.+}} : <i32>
  // CHECK: [[HI:%.+]] = comb.extract [[CUR]] from 2 : (i32) -> i30
  // CHECK: [[LO:%.+]] = comb.extract [[CUR]] from 0 : (i32) -> i1
  // CHECK: [[NEW:%.+]] = comb.concat [[HI]], %{{.+}}, [[LO]] : i30, i1, i1
  // CHECK: arc.state_write %{{.+}} = [[NEW]]
  llhd.drv %slice, %v after %t : i1
}
