// RUN: circt-opt %s -canonicalize='top-down=true region-simplify=aggressive' | FileCheck %s

// CHECK-LABEL: @drv_folding
// CHECK-SAME: %[[SIG:.*]]: !hw.inout<i32>
// CHECK-SAME: %[[VAL:.*]]: i32
// CHECK-SAME: %[[TIME:.*]]: !llhd.time
// CHECK-SAME: %[[COND:.*]]: i1
func.func @drv_folding(%sig: !hw.inout<i32>, %val: i32, %time: !llhd.time, %cond: i1) {
  %true = hw.constant 1 : i1
  %false = hw.constant 0 : i1

  // CHECK-NEXT: llhd.drv %[[SIG]], %[[VAL]] after %[[TIME]] :
  llhd.drv %sig, %val after %time : !hw.inout<i32>
  // CHECK-NEXT: llhd.drv %[[SIG]], %[[VAL]] after %[[TIME]] if %[[COND]] :
  llhd.drv %sig, %val after %time if %cond : !hw.inout<i32>
  llhd.drv %sig, %val after %time if %false : !hw.inout<i32>
  // CHECK-NEXT: llhd.drv %[[SIG]], %[[VAL]] after %[[TIME]] :
  llhd.drv %sig, %val after %time if %true : !hw.inout<i32>

  // CHECK-NEXT: return
  return
}
