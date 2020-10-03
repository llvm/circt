// RUN: circt-opt %s -cse -canonicalize | FileCheck %s

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

// CHECK-LABEL: @drv_folding
// CHECK-SAME: %[[SIG:.*]]: !llhd.sig<i32>
// CHECK-SAME: %[[VAL:.*]]: i32
// CHECK-SAME: %[[TIME:.*]]: !llhd.time
// CHECK-SAME: %[[COND:.*]]: i1
func @drv_folding(%sig: !llhd.sig<i32>, %val: i32, %time: !llhd.time, %cond: i1) {
  %true = llhd.const 1 : i1
  %false = llhd.const 0 : i1

  // CHECK-NEXT: llhd.drv %[[SIG]], %[[VAL]] after %[[TIME]] if %[[COND]] :
  llhd.drv %sig, %val after %time if %cond : !llhd.sig<i32>
  llhd.drv %sig, %val after %time if %false : !llhd.sig<i32>
  // CHECK-NEXT: llhd.drv %[[SIG]], %[[VAL]] after %[[TIME]] :
  llhd.drv %sig, %val after %time if %true : !llhd.sig<i32>

  return
}
