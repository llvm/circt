// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

// CHECK-LABEL: func.func @BoolCastChandle
// CHECK-SAME: (%arg0: !llvm.ptr) -> i1
func.func @BoolCastChandle(%arg0: !moore.chandle) -> !moore.i1 {
  // CHECK: [[NULL:%.+]] = llvm.mlir.zero : !llvm.ptr
  // CHECK: [[BOOL:%.+]] = llvm.icmp "ne" %arg0, [[NULL]] : !llvm.ptr
  %0 = moore.bool_cast %arg0 : !moore.chandle -> !moore.i1
  // CHECK: return [[BOOL]] : i1
  return %0 : !moore.i1
}

moore.class.classdecl @BoolCastClassT {
}

// CHECK-LABEL: func.func @BoolCastClassHandle
// CHECK-SAME: (%arg0: !llvm.ptr) -> i1
func.func @BoolCastClassHandle(%arg0: !moore.class<@BoolCastClassT>) -> !moore.i1 {
  // CHECK: [[NULL:%.+]] = llvm.mlir.zero : !llvm.ptr
  // CHECK: [[BOOL:%.+]] = llvm.icmp "ne" %arg0, [[NULL]] : !llvm.ptr
  %0 = moore.bool_cast %arg0 : !moore.class<@BoolCastClassT> -> !moore.i1
  // CHECK: return [[BOOL]] : i1
  return %0 : !moore.i1
}
