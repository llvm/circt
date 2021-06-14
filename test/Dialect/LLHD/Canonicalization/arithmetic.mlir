// RUN: circt-opt %s -simple-canonicalizer | FileCheck %s

// CHECK-LABEL: @check_neg_folding
func @check_neg_folding() -> (i16) {
  // CHECK-NEXT: %[[NEG:.*]] = llhd.const -5 : i16
  %a = llhd.const 5 : i16
  %0 = llhd.neg %a : i16

  // CHECK-NEXT: return %[[NEG]] : i16
  return %0 : i16
}

// CHECK-LABEL: @check_smod_folding
// CHECK-SAME: %[[IN:.*]]: i32
func @check_smod_folding(%in : i32) -> (i32, i32, i32, i32, i32, i32, i32) {
  // CHECK-DAG: %[[FOUR:.*]] = llhd.const 4 : i32
  // CHECK-DAG: %[[ONE:.*]] = llhd.const 1 : i32
  // CHECK-DAG: %[[NEGONE:.*]] = llhd.const -1 : i32
  // CHECK-DAG: %[[NEGFOUR:.*]] = llhd.const -4 : i32
  // CHECK-DAG: %[[ZERO:.*]] = llhd.const 0 : i32
  %0 = llhd.const 5 : i32
  %1 = llhd.const -5 : i32
  %2 = llhd.const 9 : i32
  %3 = llhd.const -9 : i32
  %4 = llhd.const 1 : i32
  %5 = llhd.const 0 : i32
  // check correct calculation of smod if lhs and rhs are positive
  %6 = llhd.smod %2, %0 : i32
  // check correct calculation of smod if lhs is negative and rhs is positive
  %7 = llhd.smod %3, %0 : i32
  // check correct calculation of smod if lhs is positive and rhs is negative
  %8 = llhd.smod %2, %1 : i32
  // check correct calculation of smod if lhs and rhs are negative
  %9 = llhd.smod %3, %1 : i32
  // check correct pattern replacement if rhs is one
  %10 = llhd.smod %in, %4 : i32
  // check correct pattern replacement if lhs is zero
  %11 = llhd.smod %5, %in : i32
  // check correct pattern replacement if lhs and rhs are the same register
  %12 = llhd.smod %in, %in : i32
  // CHECK-NEXT: return %[[FOUR]], %[[ONE]], %[[NEGONE]], %[[NEGFOUR]], %[[ZERO]], %[[ZERO]], %[[ZERO]] : i32, i32, i32, i32, i32, i32, i32
  return %6, %7, %8, %9, %10, %11, %12 : i32, i32, i32, i32, i32, i32, i32
}

// CHECK-LABEL: @check_eq_folding
// CHECK-SAME: %[[VAL_0:.*]]: i64,
// CHECK-SAME: %[[VAL_1:.*]]: i1,
// CHECK-SAME: %[[VAL_2:.*]]: tuple<i1, i2, i3>
func @check_eq_folding(%a : i64, %b : i1, %tup : tuple<i1, i2, i3>) -> (i1, i1, i1, i1) {
  %c1 = llhd.const 1 : i1
  %c3 = llhd.const 3 : i64
  %c4 = llhd.const 4 : i64
  %0 = llhd.eq %b, %c1 : i1
  // CHECK-DAG: %[[VAL_3:.*]] = llhd.const true : i1
  %1 = llhd.eq %a, %a : i64
  %2 = llhd.eq %tup, %tup : tuple<i1, i2, i3>
  // CHECK-DAG: %[[VAL_4:.*]] = llhd.const false : i1
  %3 = llhd.eq %c3, %c4 : i64

  // CHECK-NEXT: return %[[VAL_1]], %[[VAL_3]], %[[VAL_3]], %[[VAL_4]] : i1, i1, i1, i1
  return %0, %1, %2, %3 : i1, i1, i1, i1
}

// CHECK-LABEL: @check_eq_patterns
// CHECK-SAME: %[[A:.*]]: i1, %[[B:.*]]: i1, %[[C:.*]]: i4, %[[D:.*]]: i4
func @check_eq_patterns(%a : i1, %b : i1, %c : i4, %d : i4) -> (i1, i1) {
  // CHECK-NEXT: %[[XOR:.*]] = llhd.xor %[[A]], %[[B]] : i1
  // CHECK-NEXT: %[[NOT:.*]] = llhd.not %[[XOR]] : i1
  %0 = llhd.eq %a, %b : i1
  // CHECK-NEXT: %[[EQ:.*]] = llhd.eq %[[C]], %[[D]] : i4
  %1 = llhd.eq %c, %d : i4

  // CHECK-NEXT: return %[[NOT]], %[[EQ]] : i1, i1
  return %0, %1 : i1, i1
}

// CHECK-LABEL: @check_neq_folding
// CHECK-SAME:  %[[VAL_0:.*]]: i64,
// CHECK-SAME:  %[[VAL_1:.*]]: i1,
// CHECK-SAME:  %[[VAL_2:.*]]: tuple<i1, i2, i3>
func @check_neq_folding(%a : i64, %b : i1, %tup : tuple<i1, i2, i3>) -> (i1, i1, i1, i1) {
  %c0 = llhd.const 0 : i1
  %c3 = llhd.const 3 : i64
  %c4 = llhd.const 4 : i64
  %0 = llhd.neq %b, %c0 : i1
  // CHECK-DAG: %[[VAL_3:.*]] = llhd.const false : i1
  %1 = llhd.neq %a, %a : i64
  %2 = llhd.neq %tup, %tup : tuple<i1, i2, i3>
  // CHECK-DAG: %[[VAL_4:.*]] = llhd.const true : i1
  %3 = llhd.neq %c3, %c4 : i64

  // CHECK-NEXT: return %[[VAL_1]], %[[VAL_3]], %[[VAL_3]], %[[VAL_4]] : i1, i1, i1, i1
  return %0, %1, %2, %3 : i1, i1, i1, i1
}

// CHECK-LABEL: @check_neq_patterns
// CHECK-SAME: %[[A:.*]]: i1, %[[B:.*]]: i1, %[[C:.*]]: i4, %[[D:.*]]: i4
func @check_neq_patterns(%a : i1, %b : i1, %c : i4, %d : i4) -> (i1, i1) {
  // CHECK-NEXT: %[[XOR:.*]] = llhd.xor %[[A]], %[[B]] : i1
  %0 = llhd.neq %a, %b : i1
  // CHECK-NEXT: %[[NEQ:.*]] = llhd.neq %[[C]], %[[D]] : i4
  %1 = llhd.neq %c, %d : i4

  // CHECK-NEXT: return %[[XOR]], %[[NEQ]] : i1, i1
  return %0, %1 : i1, i1
}
