// RUN: circt-opt %s --arc-lower-state | FileCheck %s

// A module-level drive of an array element
// (`llhd.drv (llhd.sig.array_get %sig[%idx])`) lowers as a read-modify-write
// of the parent signal's storage through `hw.array_inject` -- the array-typed
// sibling of the bit-slice splice.

// CHECK-LABEL: arc.model @top
hw.module @top(in %v : i4, in %idx : i1) {
  %c0_i4 = hw.constant 0 : i4
  %true = hw.constant true
  %t = llhd.constant_time <0ns, 0d, 1e>
  %init = hw.array_create %c0_i4, %c0_i4 : i4
  %sig = llhd.sig %init : !hw.array<2xi4>
  %elem = llhd.sig.array_get %sig[%true] : <!hw.array<2xi4>>
  %elemDyn = llhd.sig.array_get %sig[%idx] : <!hw.array<2xi4>>
  // CHECK-DAG: [[CUR:%.+]] = arc.state_read %{{.+}} : <!hw.array<2xi4>>
  // CHECK-DAG: [[NEW:%.+]] = hw.array_inject [[CUR]][%{{.+}}], %{{.+}} : !hw.array<2xi4>
  // CHECK: arc.state_write %{{.+}} = [[NEW]]
  llhd.drv %elem, %v after %t : i4
  // The dynamic-index element drive takes the same path.
  // CHECK: [[CUR2:%.+]] = arc.state_read %{{.+}} : <!hw.array<2xi4>>
  // CHECK: [[NEW2:%.+]] = hw.array_inject [[CUR2]][%{{.+}}], %{{.+}} : !hw.array<2xi4>
  // CHECK: arc.state_write %{{.+}} = [[NEW2]]
  llhd.drv %elemDyn, %v after %t : i4
}
