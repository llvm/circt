// RUN: circt-opt %s -canonicalize='top-down=true region-simplify=true' | FileCheck %s

// CHECK-LABEL: @const_hoisting
// CHECK-SAME: %[[SIG:.*]]: !llhd.sig<i32>
func @const_hoisting(%sig : !llhd.sig<i32>) {
  // CHECK-DAG: %[[C0:.*]] = llhd.const -1 : i32
  // CHECK-DAG: %[[TIME:.*]] = llhd.const #llhd.time<1ns, 0d, 0e> : !llhd.time
  // CHECK-NEXT: br ^[[BB:.*]]
  br ^bb1
// CHECK-NEXT: ^[[BB]]
^bb1:
  %0 = llhd.const -1 : i32
  %1 = llhd.const #llhd.time<1ns, 0d, 0e> : !llhd.time
  // CHECK-NEXT: llhd.drv %[[SIG]], %[[C0]] after %[[TIME]] : !llhd.sig<i32>
  llhd.drv %sig, %0 after %1 : !llhd.sig<i32>
  br ^bb1
}
