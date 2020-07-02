// RUN: circt-opt %s -mlir-print-op-generic -split-input-file -verify-diagnostics | circt-opt | circt-opt | FileCheck %s

// CHECK-LABEL: @exts_integers
// CHECK-SAME: %[[CI1:.*]]: i1
// CHECK-SAME: %[[CI32:.*]]: i32
func @exts_integers(%cI1 : i1, %cI32 : i32) {
    // CHECK-NEXT: %{{.*}} = llhd.exts %[[CI1]], 0 : i1 -> i1
    %0 = llhd.exts %cI1, 0 : i1 -> i1
    // CHECK-NEXT: %{{.*}} = llhd.exts %[[CI32]], 0 : i32 -> i5
    %1 = llhd.exts %cI32, 0 : i32 -> i5

    return
}

// CHECK-LABEL: @exts_signals
// CHECK-SAME: %[[SI1:.*]]: !llhd.sig<i1>
// CHECK-SAME: %[[SI32:.*]]: !llhd.sig<i32>
func @exts_signals (%sI1 : !llhd.sig<i1>, %sI32 : !llhd.sig<i32>) -> () {
    // CHECK-NEXT: %{{.*}} = llhd.exts %[[SI1]], 0 : !llhd.sig<i1> -> !llhd.sig<i1>
    %0 = llhd.exts %sI1, 0 : !llhd.sig<i1> -> !llhd.sig<i1>
    // CHECK-NEXT: %{{.*}} = llhd.exts %[[SI32]], 0 : !llhd.sig<i32> -> !llhd.sig<i5>
    %1 = llhd.exts %sI32, 0 : !llhd.sig<i32> -> !llhd.sig<i5>

    return
}

// CHECK-LABEL: @dexts_integers
// CHECK-SAME: %[[CI1:.*]]: i1,
// CHECK-SAME: %[[CI32:.*]]: i32,
// CHECK-SAME: %[[IND0:.*]]: i5,
// CHECK-SAME: %[[IND1:.*]]: i10
func @dexts_integers(%cI1 : i1, %cI32 : i32, %i0 : i5, %i1 : i10) {
    // CHECK-NEXT: %{{.*}} = llhd.dexts %[[CI1]], %[[IND0]] : (i1, i5) -> i1
    %0 = llhd.dexts %cI1, %i0 : (i1, i5) -> i1
    // CHECK-NEXT: %{{.*}} = llhd.dexts %[[CI32]], %[[IND1]] : (i32, i10) -> i15
    %1 = llhd.dexts %cI32, %i1 : (i32, i10) -> i15

    return
}

// CHECK-LABEL: @dexts_signals
// CHECK-SAME: %[[SI1:.*]]: !llhd.sig<i1>,
// CHECK-SAME: %[[SI32:.*]]: !llhd.sig<i32>,
// CHECK-SAME: %[[IND0:.*]]: i5,
// CHECK-SAME: %[[IND1:.*]]: i10
func @dexts_signals (%sI1 : !llhd.sig<i1>, %sI32 : !llhd.sig<i32>, %i0 : i5, %i1 : i10) {
    // CHECK-NEXT: %{{.*}} = llhd.dexts %[[SI1]], %[[IND0]] : (!llhd.sig<i1>, i5) -> !llhd.sig<i1>
    %0 = llhd.dexts %sI1, %i0 : (!llhd.sig<i1>, i5) -> !llhd.sig<i1>
    // CHECK-NEXT: %{{.*}} = llhd.dexts %[[SI32]], %[[IND1]] : (!llhd.sig<i32>, i10) -> !llhd.sig<i5>
    %1 = llhd.dexts %sI32, %i1 : (!llhd.sig<i32>, i10) -> !llhd.sig<i5>

    return
}

// CHECK-LABEL: @dexts_vec
// CHECK-SAME: %[[V1:.*]]: vector<1xi1>,
// CHECK-SAME: %[[V10:.*]]: vector<10xi1>,
// CHECK-SAME: %[[IND0:.*]]: i5,
// CHECK-SAME: %[[IND1:.*]]: i10
func @dexts_vec(%v1 : vector<1xi1>, %v10 : vector<10xi1>, %i0 : i5, %i1 : i10) {
    // CHECK-NEXT: %{{.*}} = llhd.dexts %[[V1]], %[[IND0]] : (vector<1xi1>, i5) -> vector<1xi1>
    %0 = llhd.dexts %v1, %i0 : (vector<1xi1>, i5) -> vector<1xi1>
    // CHECK-NEXT: %{{.*}} = llhd.dexts %[[V10]], %[[IND1]] : (vector<10xi1>, i10) -> vector<5xi1>
    %1 = llhd.dexts %v10, %i1 : (vector<10xi1>, i10) -> vector<5xi1>

    return
}

// -----

func @illegal_int_to_sig(%c : i32) {
    // expected-error @+1 {{failed to verify that 'target' and 'result' have to be both either signless integers, signals or vectors with the same element type}}
    %0 = llhd.exts %c, 0 : i32 -> !llhd.sig<i10>

    return
}

// -----

func @illegal_sig_to_int(%s : !llhd.sig<i32>) {
    // expected-error @+1 {{failed to verify that 'target' and 'result' have to be both either signless integers, signals or vectors with the same element type}}
    %0 = llhd.exts %s, 0 : !llhd.sig<i32> -> i10

    return
}

// -----

func @illegal_out_too_big(%c : i32) {
    // expected-error @+1 {{failed to verify that 'start' + size of the slice have to be smaller or equal to the 'target' size}}
    %0 = llhd.exts %c, 0 : i32 -> i40

    return
}

// -----

func @dexts_illegal_conversion(%s : !llhd.sig<i32>, %i : i1) {
    // expected-error @+1 {{'llhd.dexts' op failed to verify that 'target' and 'result' types have to match apart from their width}}
    %0 = llhd.dexts %s, %i : (!llhd.sig<i32>, i1) -> i10

    return
}

// -----

func @dexts_illegal_out_too_wide(%c : i32, %i : i1) {
    // expected-error @+1 {{'llhd.dexts' op failed to verify that the result width cannot be larger than the target operand width}}
    %0 = llhd.dexts %c, %i : (i32, i1) -> i40

    return
}

// -----

func @dexts_illegal_vec_element_conversion(%c : vector<1xi1>, %i : i1) {
    // expected-error @+1 {{'llhd.dexts' op failed to verify that 'target' and 'result' types have to match apart from their width}}
    %0 = llhd.dexts %c, %i : (vector<1xi1>, i1) -> vector<1xi10>

    return
}
