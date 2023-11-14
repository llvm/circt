// RUN: circt-opt %s --convert-llhd-to-llvm | FileCheck %s

// CHECK-LABEL: llvm.func @lower_var(
// CHECK-SAME:    %arg0: i1, %arg1: i32) {
func.func @lower_var(%arg0: i1, %arg1: i32) {
  // CHECK: [[VAR0:%.+]] = llvm.alloca {{%.+}} x i1
  // CHECK: llvm.store %arg0, [[VAR0]]
  %0 = llhd.var %arg0 : i1
  // CHECK: [[VAR1:%.+]] = llvm.alloca {{%.+}} x i32
  // CHECK: llvm.store %arg1, [[VAR1]]
  %1 = llhd.var %arg1 : i32
  // CHECK: llvm.return
  return
}

// CHECK-LABEL: llvm.func @lower_load(
// CHECK-SAME:    %arg0: !llvm.ptr, %arg1: !llvm.ptr) {
func.func @lower_load(%arg0: !llhd.ptr<i1>, %arg1: !llhd.ptr<i32>) {
  // CHECK: llvm.load %arg0 : !llvm.ptr -> i1
  %0 = llhd.load %arg0 : !llhd.ptr<i1>
  // CHECK: llvm.load %arg1 : !llvm.ptr -> i32
  %1 = llhd.load %arg1 : !llhd.ptr<i32>
  // CHECK: llvm.return
  return
}

// CHECK-LABEL: llvm.func @lower_store(
// CHECK-SAME:    %arg0: i1, %arg1: !llvm.ptr, %arg2: i32, %arg3: !llvm.ptr) {
func.func @lower_store(%arg0: i1, %arg1: !llhd.ptr<i1>, %arg2: i32, %arg3: !llhd.ptr<i32>) {
  // CHECK: llvm.store %arg0, %arg1 : i1, !llvm.ptr
  llhd.store %arg1, %arg0 : !llhd.ptr<i1>
  // CHECK: llvm.store %arg2, %arg3 : i32, !llvm.ptr
  llhd.store %arg3, %arg2 : !llhd.ptr<i32>
  // CHECK: llvm.return
  return
}
