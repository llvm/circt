// RUN: circt-opt -canonicalize='top-down=true region-simplify=true' %s | FileCheck %s

// CHECK-LABEL: @check_shl_folding
// CHECK-SAME: %[[BASE:.*]]: i4
// CHECK-SAME: %[[HIDDEN:.*]]: i8
func @check_shl_folding(%base : i4, %hidden : i8) -> (i4, i4) {
  // CHECK-NEXT: %[[NEGSEVEN:.*]] = llhd.const -7 : i4
  %0 = llhd.const 2 : i4
  %1 = llhd.const 4 : i4
  %2 = llhd.const 0 : i4
  %3 = llhd.shl %0, %1, %0 : (i4, i4, i4) -> i4
  // check correct pattern replacement if amount is constant zero
  %4 = llhd.shl %base, %hidden, %2 : (i4, i8, i4) -> i4
  // CHECK-NEXT: return %[[NEGSEVEN]], %[[BASE]] : i4, i4
  return %3, %4 : i4, i4
}

// CHECK-LABEL: @check_shr_folding
// CHECK-SAME: %[[BASE:.*]]: i4
// CHECK-SAME: %[[HIDDEN:.*]]: i8
func @check_shr_folding(%base : i4, %hidden : i8) -> (i4, i4) {
  // CHECK-NEXT: %[[NEGSEVEN:.*]] = llhd.const -7 : i4
  %0 = llhd.const 2 : i4
  %1 = llhd.const 4 : i4
  %2 = llhd.const 0 : i4
  %3 = llhd.shr %1, %0, %0 : (i4, i4, i4) -> i4
  // check correct pattern replacement if amount is constant zero
  %4 = llhd.shr %base, %hidden, %2 : (i4, i8, i4) -> i4
  // CHECK-NEXT: return %[[NEGSEVEN]], %[[BASE]] : i4, i4
  return %3, %4 : i4, i4
}
