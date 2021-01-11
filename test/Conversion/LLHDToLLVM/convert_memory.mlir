// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py
// RUN: circt-opt %s --convert-llhd-to-llvm | FileCheck %s

// CHECK-LABEL:   llvm.func @lower_var(
// CHECK-SAME:                         %[[VAL_0:.*]]: i1,
// CHECK-SAME:                         %[[VAL_1:.*]]: i32) {
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_3:.*]] = llvm.alloca %[[VAL_2]] x i1 {alignment = 4 : i64} : (i32) -> !llvm.ptr<i1>
// CHECK:           llvm.store %[[VAL_0]], %[[VAL_3]] : !llvm.ptr<i1>
// CHECK:           %[[VAL_4:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_5:.*]] = llvm.alloca %[[VAL_4]] x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr<i32>
// CHECK:           llvm.store %[[VAL_1]], %[[VAL_5]] : !llvm.ptr<i32>
// CHECK:           llvm.return
// CHECK:         }
func @lower_var(%i1 : i1, %i32 : i32) {
  %0 = llhd.var %i1 : i1
  %1 = llhd.var %i32 : i32
  return
}

// CHECK-LABEL:   llvm.func @lower_load(
// CHECK-SAME:                          %[[VAL_0:.*]]: !llvm.ptr<i1>,
// CHECK-SAME:                          %[[VAL_1:.*]]: !llvm.ptr<i32>) {
// CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_0]] : !llvm.ptr<i1>
// CHECK:           %[[VAL_3:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr<i32>
// CHECK:           llvm.return
// CHECK:         }
func @lower_load(%i1 : !llhd.ptr<i1>, %i32 : !llhd.ptr<i32>) {
  %0 = llhd.load %i1 : !llhd.ptr<i1>
  %1 = llhd.load %i32 : !llhd.ptr<i32>
  return
}

// CHECK-LABEL:   llvm.func @lower_store(
// CHECK-SAME:                           %[[VAL_0:.*]]: i1,
// CHECK-SAME:                           %[[VAL_1:.*]]: !llvm.ptr<i1>,
// CHECK-SAME:                           %[[VAL_2:.*]]: i32,
// CHECK-SAME:                           %[[VAL_3:.*]]: !llvm.ptr<i32>) {
// CHECK:           llvm.store %[[VAL_0]], %[[VAL_1]] : !llvm.ptr<i1>
// CHECK:           llvm.store %[[VAL_2]], %[[VAL_3]] : !llvm.ptr<i32>
// CHECK:           llvm.return
// CHECK:         }
func @lower_store(%i1 : i1, %i1Ptr : !llhd.ptr<i1>, %i32 : i32, %i32Ptr : !llhd.ptr<i32>) {
  llhd.store %i1Ptr, %i1 : !llhd.ptr<i1>
  llhd.store %i32Ptr, %i32 : !llhd.ptr<i32>
  return
}
