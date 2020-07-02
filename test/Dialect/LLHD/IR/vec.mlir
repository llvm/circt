// RUN: circt-opt %s -mlir-print-op-generic -split-input-file -verify-diagnostics | circt-opt | circt-opt | FileCheck %s

// CHECK-LABEL: @check_vec
// CHECK-SAME: %[[C1:.*]]: i1, %[[C2:.*]]: i1, %[[C3:.*]]: i1
// CHECK-SAME: %[[C4:.*]]: i32, %[[C5:.*]]: i32, %[[C6:.*]]: i32
func @check_vec(%c1 : i1, %c2 : i1, %c3 : i1, %c4 : i32, %c5 : i32, %c6 : i32) {
    // CHECK-NEXT: %{{.*}} = llhd.vec %[[C1]], %[[C2]], %[[C3]] : vector<3xi1>
    %0 = llhd.vec %c1, %c2, %c3 : vector<3xi1>
    // CHECK-NEXT: %{{.*}} = llhd.vec %[[C4]], %[[C5]], %[[C6]] : vector<3xi32>
    %1 = llhd.vec %c4, %c5, %c6 : vector<3xi32>

    return
}

// TODO: add more tests
