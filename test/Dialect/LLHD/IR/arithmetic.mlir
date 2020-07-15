// RUN: circt-opt %s -mlir-print-op-generic | circt-opt | circt-opt | FileCheck %s

// CHECK-LABEL: @check_arithmetic
// CHECK-SAME: %[[A:.*]]: i64
// CHECK-SAME: %[[VEC:.*]]: vector<3xi32>
// CHECK-SAME: %[[TUP:.*]]: tuple<i1, i2, i3>
func @check_arithmetic(%a : i64, %vec : vector<3xi32>, %tup : tuple<i1, i2, i3>) -> () {
    // CHECK-NEXT: %{{.*}} = llhd.neg %[[A]] : i64
    %0 = llhd.neg %a : i64

    // CHECK-NEXT: %{{.*}} = llhd.smod %[[A]], %[[A]] : i64
    %1 = llhd.smod %a, %a : i64

    // CHECK-NEXT: %{{.*}} = llhd.neq %[[A]], %[[A]] : i64
    %2 = llhd.neq %a, %a : i64
    // CHECK-NEXT: %{{.*}} = llhd.neq %[[VEC]], %[[VEC]] : vector<3xi32>
    %3 = llhd.neq %vec, %vec : vector<3xi32>
    // CHECK-NEXT: %{{.*}} = llhd.neq %[[TUP]], %[[TUP]] : tuple<i1, i2, i3>
    %4 = llhd.neq %tup, %tup : tuple<i1, i2, i3>

    // CHECK-NEXT: %{{.*}} = llhd.eq %[[A]], %[[A]] : i64
    %5 = llhd.eq %a, %a : i64
    // CHECK-NEXT: %{{.*}} = llhd.eq %[[VEC]], %[[VEC]] : vector<3xi32>
    %6 = llhd.eq %vec, %vec : vector<3xi32>
    // CHECK-NEXT: %{{.*}} = llhd.eq %[[TUP]], %[[TUP]] : tuple<i1, i2, i3>
    %7 = llhd.eq %tup, %tup : tuple<i1, i2, i3>

    return
}
