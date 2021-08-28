// RUN: circt-opt %s -mlir-print-op-generic | circt-opt | circt-opt | FileCheck %s

// CHECK-LABEL: @check_arithmetic
// CHECK-SAME: %[[A:.*]]: i64
// CHECK-SAME: %[[ARR:.*]]: !llhd.array<3xi32>
// CHECK-SAME: %[[TUP:.*]]: tuple<i1, i2, i3>
func @check_arithmetic(%a : i64, %array : !llhd.array<3xi32>, %tup : tuple<i1, i2, i3>) -> () {
  // CHECK-NEXT: %{{.*}} = llhd.neg %[[A]] : i64
  %0 = llhd.neg %a : i64

  // CHECK-NEXT: %{{.*}} = llhd.neq %[[A]], %[[A]] : i64
  %2 = llhd.neq %a, %a : i64
  // CHECK-NEXT: %{{.*}} = llhd.neq %[[ARR]], %[[ARR]] : !llhd.array<3xi32>
  %3 = llhd.neq %array, %array : !llhd.array<3xi32>
  // CHECK-NEXT: %{{.*}} = llhd.neq %[[TUP]], %[[TUP]] : tuple<i1, i2, i3>
  %4 = llhd.neq %tup, %tup : tuple<i1, i2, i3>

  // CHECK-NEXT: %{{.*}} = llhd.eq %[[A]], %[[A]] : i64
  %5 = llhd.eq %a, %a : i64
  // CHECK-NEXT: %{{.*}} = llhd.eq %[[ARR]], %[[ARR]] : !llhd.array<3xi32>
  %6 = llhd.eq %array, %array : !llhd.array<3xi32>
  // CHECK-NEXT: %{{.*}} = llhd.eq %[[TUP]], %[[TUP]] : tuple<i1, i2, i3>
  %7 = llhd.eq %tup, %tup : tuple<i1, i2, i3>

  return
}
