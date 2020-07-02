// RUN: circt-opt %s -mlir-print-op-generic -split-input-file -verify-diagnostics | circt-opt | circt-opt | FileCheck %s

// CHECK-LABEL: @check_tuple
// CHECK-SAME: %[[C1:.*]]: i1
// CHECK-SAME: %[[C2:.*]]: i2
// CHECK-SAME: %[[C3:.*]]: i3
// CHECK-SAME: %[[VEC:.*]]: vector<3xi32>
// CHECK-SAME: %[[TUP:.*]]: tuple<i8, i32, i16>
func @check_tuple(%c1 : i1, %c2 : i2, %c3 : i3, %vec : vector<3xi32>, %tup : tuple<i8, i32, i16>) {
    // CHECK-NEXT: %{{.*}} = llhd.tuple : tuple<>
    %0 = llhd.tuple : tuple<>
    // CHECK-NEXT: %{{.*}} = llhd.tuple %[[C1]] : tuple<i1>
    %1 = llhd.tuple %c1 : tuple<i1>
    // CHECK-NEXT: %{{.*}} = llhd.tuple %[[C1]], %[[C2]], %[[C3]] : tuple<i1, i2, i3>
    %2 = llhd.tuple %c1, %c2, %c3 : tuple<i1, i2, i3>
    // CHECK-NEXT: %{{.*}} = llhd.tuple %[[C1]], %[[VEC]], %[[TUP]] : tuple<i1, vector<3xi32>, tuple<i8, i32, i16>
    %3 = llhd.tuple %c1, %vec, %tup : tuple<i1, vector<3xi32>, tuple<i8, i32, i16>>

    return
}

// TODO: add verification tests
