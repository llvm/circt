// RUN: circt-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL: @drv_folding
// CHECK-SAME: %[[SIG:.*]]: !llhd.sig<i32>
// CHECK-SAME: %[[VAL:.*]]: i32
// CHECK-SAME: %[[TIME:.*]]: !llhd.time
// CHECK-SAME: %[[COND:.*]]: i1
func @drv_folding(%sig: !llhd.sig<i32>, %val: i32, %time: !llhd.time, %cond: i1) {
  %true = llhd.const 1 : i1
  %false = llhd.const 0 : i1

  // CHECK-NEXT: llhd.drv %[[SIG]], %[[VAL]] after %[[TIME]] :
  llhd.drv %sig, %val after %time : !llhd.sig<i32>
  // CHECK-NEXT: llhd.drv %[[SIG]], %[[VAL]] after %[[TIME]] if %[[COND]] :
  llhd.drv %sig, %val after %time if %cond : !llhd.sig<i32>
  llhd.drv %sig, %val after %time if %false : !llhd.sig<i32>
  // CHECK-NEXT: llhd.drv %[[SIG]], %[[VAL]] after %[[TIME]] :
  llhd.drv %sig, %val after %time if %true : !llhd.sig<i32>

  // CHECK-NEXT: return
  return
}

// CHECK-LABEL: @sig_folding
llhd.entity @sig_folding() -> () {
  %false = llhd.const 0 : i1
  // CHECK: llhd.sig "sig1"
  // CHECK-NOT: llhd.sig
  %0 = llhd.sig "sig1" %false : i1
  %1 = llhd.prb %0 : !llhd.sig<i1>
  %2 = llhd.sig "sig2" %1 : i1
}
