// RUN: circt-opt %s -mlir-print-op-generic | circt-opt | circt-opt | FileCheck %s

// CHECK-LABEL: @check_arithmetic
// CHECK-SAME: %[[A:.*]]: i64
// CHECK-SAME: %[[ARR:.*]]: !hw.array<3xi32>
// CHECK-SAME: %[[TUP:.*]]: !hw.struct<foo: i1, bar: i2, baz: i3>
func @check_arithmetic(%a : i64, %array : !hw.array<3xi32>, %tup : !hw.struct<foo: i1, bar: i2, baz: i3>) -> () {
  // CHECK-NEXT: %{{.*}} = llhd.neq %[[A]], %[[A]] : i64
  %2 = llhd.neq %a, %a : i64
  // CHECK-NEXT: %{{.*}} = llhd.neq %[[ARR]], %[[ARR]] : !hw.array<3xi32>
  %3 = llhd.neq %array, %array : !hw.array<3xi32>
  // CHECK-NEXT: %{{.*}} = llhd.neq %[[TUP]], %[[TUP]] : !hw.struct<foo: i1, bar: i2, baz: i3>
  %4 = llhd.neq %tup, %tup : !hw.struct<foo: i1, bar: i2, baz: i3>

  // CHECK-NEXT: %{{.*}} = llhd.eq %[[A]], %[[A]] : i64
  %5 = llhd.eq %a, %a : i64
  // CHECK-NEXT: %{{.*}} = llhd.eq %[[ARR]], %[[ARR]] : !hw.array<3xi32>
  %6 = llhd.eq %array, %array : !hw.array<3xi32>
  // CHECK-NEXT: %{{.*}} = llhd.eq %[[TUP]], %[[TUP]] : !hw.struct<foo: i1, bar: i2, baz: i3>
  %7 = llhd.eq %tup, %tup : !hw.struct<foo: i1, bar: i2, baz: i3>

  return
}
