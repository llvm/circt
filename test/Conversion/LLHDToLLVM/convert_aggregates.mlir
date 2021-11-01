// RUN: circt-opt %s --convert-llhd-to-llvm | FileCheck %s

// CHECK-LABEL: llvm.func @convertBitcast
func @convertBitcast(%arg0 : i32, %arg1: !hw.array<2xi32>, %arg2: !hw.struct<foo: i32, bar: i32>) {

  // CHECK-NEXT: %[[ONE1:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: %[[A1:.*]] = llvm.alloca %[[ONE1]] x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr<i32>
  // CHECK-NEXT: llvm.store %arg0, %[[A1]] : !llvm.ptr<i32>
  // CHECK-NEXT: %[[B1:.*]] = llvm.bitcast %[[A1]] : !llvm.ptr<i32> to !llvm.ptr<array<4 x i8>>
  // CHECK-NEXT: llvm.load %[[B1]] : !llvm.ptr<array<4 x i8>>
  %0 = hw.bitcast %arg0 : (i32) -> !hw.array<4xi8>

// CHECK-NEXT: %[[ONE2:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[A2:.*]] = llvm.alloca %[[ONE2]] x !llvm.array<2 x i32> {alignment = 4 : i64} : (i32) -> !llvm.ptr<array<2 x i32>>
// CHECK-NEXT: llvm.store %arg1, %[[A2]] : !llvm.ptr<array<2 x i32>>
// CHECK-NEXT: %[[B2:.*]] = llvm.bitcast %[[A2]] : !llvm.ptr<array<2 x i32>> to !llvm.ptr<i64>
// CHECK-NEXT: llvm.load %[[B2]] : !llvm.ptr<i64>
  %1 = hw.bitcast %arg1 : (!hw.array<2xi32>) -> i64

// CHECK-NEXT: %[[ONE3:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[A3:.*]] = llvm.alloca %[[ONE3]] x !llvm.struct<(i32, i32)> {alignment = 4 : i64} : (i32) -> !llvm.ptr<struct<(i32, i32)>>
// CHECK-NEXT: llvm.store %arg2, %[[A3]] : !llvm.ptr<struct<(i32, i32)>>
// CHECK-NEXT: %[[B3:.*]] = llvm.bitcast %[[A3]] : !llvm.ptr<struct<(i32, i32)>> to !llvm.ptr<i64>
// CHECK-NEXT: llvm.load %[[B3]] : !llvm.ptr<i64>
  %2 = hw.bitcast %arg2 : (!hw.struct<foo: i32, bar: i32>) -> i64

  return
}

// CHECK-LABEL: llvm.func @convertArray
func @convertArray(%arg0 : i1, %arg1: !hw.array<2xi32>) {

  // CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-NEXT: %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: %[[ALLOCA:.*]] = llvm.alloca %[[ONE]] x !llvm.array<2 x i32> {alignment = 4 : i64} : (i32) -> !llvm.ptr<array<2 x i32>>
  // CHECK-NEXT: llvm.store %arg1, %[[ALLOCA]] : !llvm.ptr<array<2 x i32>>
  // CHECK-NEXT: %[[ZEXT:.*]] = llvm.zext %arg0 : i1 to i2
  // CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[ALLOCA]][%[[ZERO]], %[[ZEXT]]] : (!llvm.ptr<array<2 x i32>>, i32, i2) -> !llvm.ptr<i32>
  // CHECK-NEXT: llvm.load %[[GEP]] : !llvm.ptr<i32>
  %0 = hw.array_get %arg1[%arg0] : !hw.array<2xi32>

  // CHECK-NEXT: %6 = llvm.mlir.constant(0 : i32) : i32
  // CHECK-NEXT: %7 = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: %8 = llvm.alloca %7 x !llvm.array<2 x i32> {alignment = 4 : i64} : (i32) -> !llvm.ptr<array<2 x i32>>
  // CHECK-NEXT: llvm.store %arg1, %8 : !llvm.ptr<array<2 x i32>>
  // CHECK-NEXT: %9 = llvm.zext %arg0 : i1 to i2
  // CHECK-NEXT: %10 = llvm.getelementptr %8[%6, %9] : (!llvm.ptr<array<2 x i32>>, i32, i2) -> !llvm.ptr<i32>
  // CHECK-NEXT: %11 = llvm.bitcast %10 : !llvm.ptr<i32> to !llvm.ptr<array<1 x i32>>
  // CHECK-NEXT: llvm.load %11 : !llvm.ptr<array<1 x i32>>
  %1 = hw.array_slice %arg1 at %arg0 : (!hw.array<2xi32>) -> !hw.array<1xi32>

  // CHECK-NEXT: %[[UNDEF:.*]] = llvm.mlir.undef : !llvm.array<4 x i32>
  // CHECK-NEXT: %[[E1:.*]] = llvm.extractvalue %arg1[0 : i32] : !llvm.array<2 x i32>
  // CHECK-NEXT: %[[I1:.*]] = llvm.insertvalue %[[E1]], %[[UNDEF]][0 : i32] : !llvm.array<4 x i32>
  // CHECK-NEXT: %[[E2:.*]] = llvm.extractvalue %arg1[1 : i32] : !llvm.array<2 x i32>
  // CHECK-NEXT: %[[I2:.*]] = llvm.insertvalue %[[E2]], %[[I1]][1 : i32] : !llvm.array<4 x i32>
  // CHECK-NEXT: %[[E3:.*]] = llvm.extractvalue %arg1[0 : i32] : !llvm.array<2 x i32>
  // CHECK-NEXT: %[[I3:.*]] = llvm.insertvalue %[[E3]], %[[I2]][2 : i32] : !llvm.array<4 x i32>
  // CHECK-NEXT: %[[E4:.*]] = llvm.extractvalue %arg1[1 : i32] : !llvm.array<2 x i32>
  // CHECK-NEXT: llvm.insertvalue %[[E4]], %[[I3]][3 : i32] : !llvm.array<4 x i32>
  %2 = hw.array_concat %arg1, %arg1 : !hw.array<2xi32>, !hw.array<2xi32>
  
  return
}

// CHECK-LABEL: llvm.func @convertStruct
func @convertStruct(%arg0 : i32, %arg1: !hw.struct<foo: i32, bar: i8>) {
  // CHECK-NEXT: llvm.extractvalue %arg1[1 : i32] : !llvm.struct<(i8, i32)>
  %0 = hw.struct_extract %arg1["foo"] : !hw.struct<foo: i32, bar: i8>

  // CHECK-NEXT: llvm.insertvalue %arg0, %arg1[1 : i32] : !llvm.struct<(i8, i32)>
  %1 = hw.struct_inject %arg1["foo"], %arg0 : !hw.struct<foo: i32, bar: i8>

  return
}
