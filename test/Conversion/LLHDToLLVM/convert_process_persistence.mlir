// RUN: circt-opt %s --convert-llhd-to-llvm | FileCheck %s

// CHECK: @dummyA
// CHECK: @dummyB
func.func private @dummyA(%arg0: i42)
func.func private @dummyB(%arg0: !llhd.ptr<i42>)

// CHECK-LABEL: llvm.func @PersistValuesAcrossPotentialResumePoints(
// CHECK-SAME:    %arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr)
llhd.proc @PersistValuesAcrossPotentialResumePoints () -> () {
  // Values used across basic blocks get persisted directly
  // CHECK: [[TMP1:%.+]] = llvm.mlir.constant(1337 :
  // CHECK: [[PERSIST_PTR1:%.+]] = llvm.getelementptr %arg1[0, 3, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, ptr, struct<(i42, ptr)>)>
  // CHECK: llvm.store [[TMP1]], [[PERSIST_PTR1]] : i42, !llvm.ptr
  %0 = hw.constant 1337 : i42

  // Variables used across basic blocks get persisted by loading their value
  // CHECK: [[TMP2:%.+]] = llvm.alloca {{%.+}} x i42
  // CHECK: llvm.store [[TMP1]], [[TMP2]] : i42, !llvm.ptr
  // CHECK: [[PERSIST_PTR2:%.+]] = llvm.getelementptr %arg1[0, 3, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, ptr, struct<(i42, ptr)>)>
  // CHECK: [[TMP3:%.+]] = llvm.load [[TMP2]] : !llvm.ptr -> i42
  // CHECK: llvm.store [[TMP3]], [[PERSIST_PTR2]] : i42, !llvm.ptr
  %1 = llhd.var %0 : i42

  // CHECK: llvm.br [[BB:\^.+]]
  cf.br ^resume

  // CHECK: [[BB]]:
^resume:
  // CHECK: [[PERSIST_PTR2:%.+]] = llvm.getelementptr %arg1[0, 3, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, ptr, struct<(i42, ptr)>)>
  // CHECK: [[PERSIST_PTR1:%.+]] = llvm.getelementptr %arg1[0, 3, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, ptr, struct<(i42, ptr)>)>
  // CHECK: [[TMP1:%.+]] = llvm.load [[PERSIST_PTR1]] : !llvm.ptr -> i42
  // CHECK: llvm.call @dummyA([[TMP1]])
  // CHECK: llvm.call @dummyB([[PERSIST_PTR2]])
  func.call @dummyA(%0) : (i42) -> ()
  func.call @dummyB(%1) : (!llhd.ptr<i42>) -> ()

  // CHECK: llvm.br [[BB]]
  cf.br ^resume
}
