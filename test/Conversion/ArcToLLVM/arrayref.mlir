// RUN: circt-opt %s --lower-arc-to-llvm | FileCheck %s

// CHECK-LABEL: @Types
// CHECK-SAME: %arg0: !llvm.ptr {llvm.dereferenceable = 8 : i64}
// CHECK-SAME: -> !llvm.ptr
func.func @Types(%arg0: !arc.arrayref<2xi32>) -> !arc.arrayref<2xi32> {
  return %arg0 : !arc.arrayref<2xi32>
}

// CHECK-LABEL: @StorageGet
func.func @StorageGet(%arg0: !arc.storage<12>) -> !arc.state<!arc.arrayref<2xi32>> {
  %0 = arc.storage.get %arg0[4] : !arc.storage<12> -> !arc.state<!arc.arrayref<2xi32>>
  return %0 : !arc.state<!arc.arrayref<2xi32>>
}

// CHECK-LABEL: @StateRead
// CHECK-NEXT: return %arg0
func.func @StateRead(%arg0: !arc.state<!arc.arrayref<2xi32>>) -> !arc.arrayref<2xi32> {
  %0 = arc.state_read %arg0 : <!arc.arrayref<2xi32>>
  return %0 : !arc.arrayref<2xi32>
}

// CHECK-LABEL: @StateWrite
// CHECK-NEXT: %[[C:.*]] = llvm.mlir.constant(8 : i64)
// CHECK-NEXT: "llvm.intr.memcpy"(%arg0, %arg1, %[[C]])
func.func @StateWrite(%arg0: !arc.state<!arc.arrayref<2xi32>>, %arg1: !arc.arrayref<2xi32>) {
  arc.state_write %arg0 = %arg1 : <!arc.arrayref<2xi32>>
  return
}

// CHECK-LABEL: @Mux
// CHECK-NEXT: %[[R:.*]] = llvm.select %arg2, %arg0, %arg1 : i1, !llvm.ptr
// CHECK-NEXT: return %[[R]]
func.func @Mux(%arg0: !arc.arrayref<2xi32>, %arg1: !arc.arrayref<2xi32>, %arg2: i1) -> !arc.arrayref<2xi32> {
  %0 = arith.select %arg2, %arg0, %arg1 : !arc.arrayref<2xi32>
  return %0 : !arc.arrayref<2xi32>
}

// CHECK-LABEL: @ArrayRefCreate
// CHECK: %[[C4:.*]] = llvm.mlir.constant(4 : i64)
// CHECK: %[[GEP0:.*]] = llvm.getelementptr %arg0[%[[C4]]]
// CHECK: llvm.store %arg1, %[[GEP0]]
// CHECK: %[[C0:.*]] = llvm.mlir.constant(0 : i64)
// CHECK: %[[GEP1:.*]] = llvm.getelementptr %arg0[%[[C0]]]
// CHECK: llvm.store %arg2, %[[GEP1]]
// CHECK: return %arg0
func.func @ArrayRefCreate(%arg0: !arc.arrayref<2xi32>, %arg1: i32, %arg2: i32) -> !arc.arrayref<2xi32> {
  %1 = arc.arrayref.create %arg0 = %arg1, %arg2 : !arc.arrayref<2xi32>
  return %1 : !arc.arrayref<2xi32>
}

// CHECK-LABEL: @ArrayRefAlloc
// CHECK-NEXT: %[[C8:.*]] = llvm.mlir.constant(8 : i64)
// CHECK-NEXT: %[[A:.*]] = llvm.alloca %[[C8]] x i8
// CHECK-NEXT: return %[[A]]
func.func @ArrayRefAlloc() -> !arc.arrayref<2xi32> {
  %0 = arc.arrayref.alloc  !arc.arrayref<2xi32>
  return %0 : !arc.arrayref<2xi32>
}

// CHECK-LABEL: @ArrayRefInit
// CHECK-NEXT: %[[C8:.*]] = llvm.mlir.constant(8 : i64)
// CHECK-NEXT: %[[A:.*]] = llvm.alloca %[[C8]] x i8
// CHECK-NEXT: %[[C4:.*]] = llvm.mlir.constant(4 : i64)
// CHECK-NEXT: %[[GEP0:.*]] = llvm.getelementptr %[[A]][%[[C4]]]
// CHECK-NEXT: %[[V123:.*]] = llvm.mlir.constant(123 : i32)
// CHECK-NEXT: llvm.store %[[V123]], %[[GEP0]]
// CHECK-NEXT: %[[C0:.*]] = llvm.mlir.constant(0 : i64)
// CHECK-NEXT: %[[GEP1:.*]] = llvm.getelementptr %[[A]][%[[C0]]]
// CHECK-NEXT: %[[V456:.*]] = llvm.mlir.constant(456 : i32)
// CHECK-NEXT: llvm.store %[[V456]], %[[GEP1]]
// CHECK-NEXT: return %[[A]]
func.func @ArrayRefInit() -> !arc.arrayref<2xi32> {
  %0 = arc.arrayref.alloc init [123 : i32, 456 : i32] !arc.arrayref<2xi32>
  return %0 : !arc.arrayref<2xi32>
}

// CHECK-LABEL: @ArrayRefInitZero
// CHECK-NEXT: %[[C8:.*]] = llvm.mlir.constant(8 : i64)
// CHECK-NEXT: %[[A:.*]] = llvm.alloca %[[C8]] x i8
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i8)
// CHECK-NEXT: "llvm.intr.memset"(%[[A]], %[[ZERO]], %[[C8]])
// CHECK-NEXT: return %[[A]]
func.func @ArrayRefInitZero() -> !arc.arrayref<2xi32> {
  %0 = arc.arrayref.alloc init [0 : i32, 0 : i32] !arc.arrayref<2xi32>
  return %0 : !arc.arrayref<2xi32>
}

// CHECK-LABEL: @ArrayRefGet
// CHECK-NEXT: %[[STRIDE:.*]] = llvm.mlir.constant(4 : i64)
// CHECK-NEXT: %[[OFF:.*]] = llvm.mul %arg1, %[[STRIDE]]
// CHECK-NEXT: %[[ADDR:.*]] = llvm.getelementptr %arg0[%[[OFF]]]
// CHECK-NEXT: %[[VAL:.*]] = llvm.load %[[ADDR]]
// CHECK-NEXT: return %[[VAL]]
func.func @ArrayRefGet(%arg0: !arc.arrayref<4xi32>, %idx: index) -> i32 {
  %0 = arc.arrayref.get %arg0[%idx] : !arc.arrayref<4xi32> -> i32
  return %0 : i32
}

// CHECK-LABEL: @ArrayRefInject
// CHECK-NEXT: %[[STRIDE:.*]] = llvm.mlir.constant(4 : i64)
// CHECK-NEXT: %[[OFF:.*]] = llvm.mul %arg1, %[[STRIDE]]
// CHECK-NEXT: %[[ADDR:.*]] = llvm.getelementptr %arg0[%[[OFF]]]
// CHECK-NEXT: llvm.store %arg2, %[[ADDR]]
// CHECK-NEXT: return %arg0
func.func @ArrayRefInject(%arg0: !arc.arrayref<4xi32>, %idx: index, %val: i32) -> !arc.arrayref<4xi32> {
  %0 = arc.arrayref.inject %arg0[%idx], %val : !arc.arrayref<4xi32>, i32 -> !arc.arrayref<4xi32>
  return %0 : !arc.arrayref<4xi32>
}

// CHECK-LABEL: @ArrayRefSlice
// CHECK-NEXT: %[[STRIDE:.*]] = llvm.mlir.constant(4 : i64)
// CHECK-NEXT: %[[OFF:.*]] = llvm.mul %arg1, %[[STRIDE]]
// CHECK-NEXT: %[[ADDR:.*]] = llvm.getelementptr %arg0[%[[OFF]]]
// CHECK-NEXT: return %[[ADDR]]
func.func @ArrayRefSlice(%arg0: !arc.arrayref<4xi32>, %lowIdx: index) -> !arc.arrayref<2xi32> {
  %0 = arc.arrayref.slice %arg0[%lowIdx] : (!arc.arrayref<4xi32>) -> !arc.arrayref<2xi32>
  return %0 : !arc.arrayref<2xi32>
}

// CHECK-LABEL: @ArrayRefCopy
// CHECK-NEXT: %[[C8:.*]] = llvm.mlir.constant(8 : i64)
// CHECK-NEXT: "llvm.intr.memcpy"(%arg0, %arg1, %[[C8]])
// CHECK-NEXT: return %arg0
func.func @ArrayRefCopy(%arg0: !arc.arrayref<2xi32>, %arg1: !arc.arrayref<2xi32>) -> !arc.arrayref<2xi32> {
  %0 = arc.arrayref.copy %arg0 = %arg1 : !arc.arrayref<2xi32>
  return %0 : !arc.arrayref<2xi32>
}
