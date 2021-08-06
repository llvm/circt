// RUN: circt-opt -canonicalize='top-down=true region-simplify=true' %s | FileCheck %s

// CHECK-LABEL: @check_not_folding
// CHECK-SAME: %[[A:.*]]: i64
func @check_not_folding(%a : i64) -> (i64, i64) {
  %c0 = llhd.const 0 : i64
  // CHECK-NEXT: %[[CN1:.*]] = llhd.const -1 : i64
  %cn1 = llhd.const -1 : i64
  %0 = llhd.not %c0 : i64
  %na = llhd.not %a : i64
  %1 = llhd.not %na : i64

  // CHECK-NEXT: return %[[CN1]], %[[A]] : i64, i64
  return %0, %1 : i64, i64
}

// CHECK-LABEL: @check_not_of_equality_patterns
// CHECK-SAME: %[[A:.*]]: i64,
// CHECK-SAME: %[[B:.*]]: i64
func @check_not_of_equality_patterns(%a : i64, %b : i64) -> (i1, i1) {
  %eq = llhd.eq %a, %b : i64
  %neq = llhd.neq %a, %b : i64
  // CHECK-NEXT: %[[NEQ:.*]] = llhd.neq %[[A]], %[[B]] : i64
  %0 = llhd.not %eq : i1
  // CHECK-NEXT: %[[EQ:.*]] = llhd.eq %[[A]], %[[B]] : i64
  %1 = llhd.not %neq : i1

  // CHECK-NEXT: return %[[NEQ]], %[[EQ]] : i1, i1
  return %0, %1 : i1, i1
}

// CHECK-LABEL: @check_and_folding
// CHECK-SAME: %[[A:.*]]: i64,
// CHECK-SAME: %[[B:.*]]: i64
func @check_and_folding(%a : i64, %b : i64) -> (i64, i64, i64, i64, i64, i64, i64) {
  // CHECK-NEXT: %[[C0:.*]] = llhd.const 0 : i64
  %c0 = llhd.const 0 : i64
  %cn1 = llhd.const -1 : i64
  %na = llhd.not %a : i64
  // CHECK-NEXT: %[[NB:.*]] = llhd.not %[[B]] : i64
  %nb = llhd.not %b : i64
  %0 = llhd.and %a, %a : i64
  %1 = llhd.and %c0, %a : i64
  %2 = llhd.and %cn1, %a : i64
  %3 = llhd.and %na, %a : i64
  %4 = llhd.and %a, %na : i64
  // CHECK-NEXT: %[[AND1:.*]] = llhd.and %[[NB]], %[[A]] : i64
  %5 = llhd.and %nb, %a : i64
  // CHECK-NEXT: %[[AND2:.*]] = llhd.and %[[A]], %[[NB]] : i64
  %6 = llhd.and %a, %nb : i64

  // CHECK-NEXT: return %[[A]], %[[C0]], %[[A]], %[[C0]], %[[C0]], %[[AND1]], %[[AND2]] : i64, i64, i64, i64, i64, i64, i64
  return %0, %1, %2, %3, %4, %5, %6 : i64, i64, i64, i64, i64, i64, i64
}

// CHECK-LABEL: @check_or_folding
// CHECK-SAME: %[[A:.*]]: i64,
// CHECK-SAME: %[[B:.*]]: i64
func @check_or_folding(%a : i64, %b : i64) -> (i64, i64, i64, i64, i64, i64, i64) {
  %c0 = llhd.const 0 : i64
  // CHECK-NEXT: %[[CN1:.*]] = llhd.const -1 : i64
  %cn1 = llhd.const -1 : i64
  %na = llhd.not %a : i64
  // CHECK-NEXT: %[[NB:.*]] = llhd.not %[[B]] : i64
  %nb = llhd.not %b : i64
  %0 = llhd.or %a, %a : i64
  %1 = llhd.or %c0, %a : i64
  %2 = llhd.or %cn1, %a : i64
  %3 = llhd.or %na, %a : i64
  %4 = llhd.or %a, %na : i64
  // CHECK-NEXT: %[[OR1:.*]] = llhd.or %[[NB]], %[[A]] : i64
  %5 = llhd.or %nb, %a : i64
  // CHECK-NEXT: %[[OR2:.*]] = llhd.or %[[A]], %[[NB]] : i64
  %6 = llhd.or %a, %nb : i64

  // CHECK-NEXT: return %[[A]], %[[A]], %[[CN1]], %[[CN1]], %[[CN1]], %[[OR1]], %[[OR2]] : i64, i64, i64, i64, i64, i64, i64
  return %0, %1, %2, %3, %4, %5, %6 : i64, i64, i64, i64, i64, i64, i64
}

// CHECK-LABEL: @check_xor_folding
// CHECK-SAME: %[[A:.*]]: i64,
// CHECK-SAME: %[[B:.*]]: i64
func @check_xor_folding(%a : i64, %b : i64) -> (i64, i64, i64, i64, i64, i64, i64) {
  // CHECK-DAG: %[[C0:.*]] = llhd.const 0 : i64
  %c0 = llhd.const 0 : i64
  // CHECK-DAG: %[[CN1:.*]] = llhd.const -1 : i64
  %cn1 = llhd.const -1 : i64
  %na = llhd.not %a : i64
  // CHECK-NEXT: %[[NB:.*]] = llhd.not %[[B]] : i64
  %nb = llhd.not %b : i64
  %0 = llhd.xor %a, %a : i64
  %1 = llhd.xor %c0, %a : i64
  // CHECK-NEXT: %[[NA:.*]] = llhd.not %[[A]] : i64
  %2 = llhd.xor %cn1, %a : i64
  %3 = llhd.xor %na, %a : i64
  %4 = llhd.xor %a, %na : i64
  // CHECK-NEXT: %[[XOR1:.*]] = llhd.xor %[[NB]], %[[A]] : i64
  %5 = llhd.xor %nb, %a : i64
  // CHECK-NEXT: %[[XOR2:.*]] = llhd.xor %[[A]], %[[NB]] : i64
  %6 = llhd.xor %a, %nb : i64

  // CHECK-NEXT: return %[[C0]], %[[A]], %[[NA]], %[[CN1]], %[[CN1]], %[[XOR1]], %[[XOR2]] : i64, i64, i64, i64, i64, i64, i64
  return %0, %1, %2, %3, %4, %5, %6 : i64, i64, i64, i64, i64, i64, i64
}

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
