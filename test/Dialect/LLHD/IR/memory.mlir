// RUN: circt-opt %s -mlir-print-op-generic -split-input-file -verify-diagnostics | circt-opt | circt-opt | FileCheck %s

// CHECK-LABEL: @check_var
// CHECK-SAME: %[[INT:.*]]: i32
// CHECK-SAME: %[[ARRAY:.*]]: !llhd.array<3xi1>
// CHECK-SAME: %[[TUP:.*]]: tuple<i1, i2, i3>
func @check_var(%int : i32, %array : !llhd.array<3xi1>, %tup : tuple<i1, i2, i3>) {
  // CHECK-NEXT: %{{.*}} = llhd.var %[[INT]] : i32
  %0 = llhd.var %int : i32
  // CHECK-NEXT: %{{.*}} = llhd.var %[[ARRAY]] : !llhd.array<3xi1>
  %1 = llhd.var %array : !llhd.array<3xi1>
  // CHECK-NEXT: %{{.*}} = llhd.var %[[TUP]] : tuple<i1, i2, i3>
  %2 = llhd.var %tup : tuple<i1, i2, i3>

  return
}

// CHECK-LABEL: @check_load
// CHECK-SAME: %[[INT:.*]]: !llhd.ptr<i32>
// CHECK-SAME: %[[ARRAY:.*]]: !llhd.ptr<!llhd.array<3xi1>>
// CHECK-SAME: %[[TUP:.*]]: !llhd.ptr<tuple<i1, i2, i3>>
func @check_load(%int : !llhd.ptr<i32>, %array : !llhd.ptr<!llhd.array<3xi1>>, %tup : !llhd.ptr<tuple<i1, i2, i3>>) {
  // CHECK-NEXT: %{{.*}} = llhd.load %[[INT]] : !llhd.ptr<i32>
  %0 = llhd.load %int : !llhd.ptr<i32>
  // CHECK-NEXT: %{{.*}} = llhd.load %[[ARRAY]] : !llhd.ptr<!llhd.array<3xi1>>
  %1 = llhd.load %array : !llhd.ptr<!llhd.array<3xi1>>
  // CHECK-NEXT: %{{.*}} = llhd.load %[[TUP]] : !llhd.ptr<tuple<i1, i2, i3>>
  %2 = llhd.load %tup : !llhd.ptr<tuple<i1, i2, i3>>

  return

}
// CHECK-LABEL: @check_store
// CHECK-SAME: %[[INT:.*]]: !llhd.ptr<i32>
// CHECK-SAME: %[[INTC:.*]]: i32
// CHECK-SAME: %[[ARRAY:.*]]: !llhd.ptr<!llhd.array<3xi1>>
// CHECK-SAME: %[[ARRAYC:.*]]: !llhd.array<3xi1>
// CHECK-SAME: %[[TUP:.*]]: !llhd.ptr<tuple<i1, i2, i3>>
// CHECK-SAME: %[[TUPC:.*]]: tuple<i1, i2, i3>
func @check_store(%int : !llhd.ptr<i32>, %intC : i32 , %array : !llhd.ptr<!llhd.array<3xi1>>, %arrayC : !llhd.array<3xi1>, %tup : !llhd.ptr<tuple<i1, i2, i3>>, %tupC : tuple<i1, i2, i3>) {
  // CHECK-NEXT: llhd.store %[[INT]], %[[INTC]] : !llhd.ptr<i32>
  llhd.store %int, %intC : !llhd.ptr<i32>
  // CHECK-NEXT: llhd.store %[[ARRAY]], %[[ARRAYC]] : !llhd.ptr<!llhd.array<3xi1>>
  llhd.store %array, %arrayC : !llhd.ptr<!llhd.array<3xi1>>
  // CHECK-NEXT: llhd.store %[[TUP]], %[[TUPC]] : !llhd.ptr<tuple<i1, i2, i3>>
  llhd.store %tup, %tupC : !llhd.ptr<tuple<i1, i2, i3>>

  return
}
