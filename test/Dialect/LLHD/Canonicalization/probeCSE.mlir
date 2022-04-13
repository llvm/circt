// RUN: circt-opt %s -cse | FileCheck %s
// XFAIL: *

// CHECK-LABEL: @check_dce_prb_but_not_cse
// CHECK-SAME: %[[SIG:.*]]: !llhd.sig<i32>
func @check_dce_prb_but_not_cse(%sig : !llhd.sig<i32>) -> (i32, i32) {
  // CHECK-NEXT: %[[P1:.*]] = llhd.prb %[[SIG]] : !llhd.sig<i32>
  %1 = llhd.prb %sig : !llhd.sig<i32>
  // CHECK-NEXT: %[[P2:.*]] = llhd.prb %[[SIG]] : !llhd.sig<i32>
  %2 = llhd.prb %sig : !llhd.sig<i32>
  %3 = llhd.prb %sig : !llhd.sig<i32>

  // CHECK-NEXT: return %[[P1]], %[[P2]]
  return %1, %2 : i32, i32
}
