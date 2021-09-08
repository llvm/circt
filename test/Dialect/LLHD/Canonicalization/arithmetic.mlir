// RUN: circt-opt %s -canonicalize='top-down=true region-simplify=true' | FileCheck %s

// CHECK-LABEL: @check_eq_folding
// CHECK-SAME: %[[VAL_0:.*]]: i64,
// CHECK-SAME: %[[VAL_1:.*]]: i1,
// CHECK-SAME: %[[VAL_2:.*]]: tuple<i1, i2, i3>
func @check_eq_folding(%a : i64, %b : i1, %tup : tuple<i1, i2, i3>) -> (i1, i1, i1, i1) {
  %c1 = hw.constant 1 : i1
  %c3 = hw.constant 3 : i64
  %c4 = hw.constant 4 : i64
  %0 = llhd.eq %b, %c1 : i1
  // CHECK-DAG: %[[VAL_3:.*]] = hw.constant true
  %1 = llhd.eq %a, %a : i64
  %2 = llhd.eq %tup, %tup : tuple<i1, i2, i3>
  // CHECK-DAG: %[[VAL_4:.*]] = hw.constant false
  %3 = llhd.eq %c3, %c4 : i64

  // CHECK-NEXT: return %[[VAL_1]], %[[VAL_3]], %[[VAL_3]], %[[VAL_4]] : i1, i1, i1, i1
  return %0, %1, %2, %3 : i1, i1, i1, i1
}

// CHECK-LABEL: @check_neq_folding
// CHECK-SAME:  %[[VAL_0:.*]]: i64,
// CHECK-SAME:  %[[VAL_1:.*]]: i1,
// CHECK-SAME:  %[[VAL_2:.*]]: tuple<i1, i2, i3>
func @check_neq_folding(%a : i64, %b : i1, %tup : tuple<i1, i2, i3>) -> (i1, i1, i1, i1) {
  %c0 = hw.constant 0 : i1
  %c3 = hw.constant 3 : i64
  %c4 = hw.constant 4 : i64
  %0 = llhd.neq %b, %c0 : i1
  // CHECK-DAG: %[[VAL_3:.*]] = hw.constant false
  %1 = llhd.neq %a, %a : i64
  %2 = llhd.neq %tup, %tup : tuple<i1, i2, i3>
  // CHECK-DAG: %[[VAL_4:.*]] = hw.constant true
  %3 = llhd.neq %c3, %c4 : i64

  // CHECK-NEXT: return %[[VAL_1]], %[[VAL_3]], %[[VAL_3]], %[[VAL_4]] : i1, i1, i1, i1
  return %0, %1, %2, %3 : i1, i1, i1, i1
}
