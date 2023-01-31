// RUN: circt-opt %s --convert-hw-to-llvm | FileCheck %s

// CHECK-LABEL: llvm.mlir.global internal @_array_global() {addr_space = 0 : i32} : !llvm.array<2 x i32> {
// CHECK-NEXT: %[[VAL_0:.*]] = llvm.mlir.undef : !llvm.array<2 x i32>
// CHECK-NEXT: %[[VAL_1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[VAL_2:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_0]][0] : !llvm.array<2 x i32>
// CHECK-NEXT: %[[VAL_3:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[VAL_4:.*]] = llvm.insertvalue %[[VAL_3]], %[[VAL_2]][1] : !llvm.array<2 x i32>
// CHECK-NEXT: llvm.return %[[VAL_4]] : !llvm.array<2 x i32>
// CHECK-NEXT: }

// CHECK-LABEL: @convertBitcast
func.func @convertBitcast(%arg0 : i32, %arg1: !hw.array<2xi32>, %arg2: !hw.struct<foo: i32, bar: i32>) {

  // CHECK: %[[AARG0:.*]] = builtin.unrealized_conversion_cast %arg0 : i32 to i32
  // CHECK-NEXT: %[[ONE1:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: %[[A1:.*]] = llvm.alloca %[[ONE1]] x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr<i32>
  // CHECK-NEXT: llvm.store %[[AARG0]], %[[A1]] : !llvm.ptr<i32>
  // CHECK-NEXT: %[[B1:.*]] = llvm.bitcast %[[A1]] : !llvm.ptr<i32> to !llvm.ptr<array<4 x i8>>
  // CHECK-NEXT: llvm.load %[[B1]] : !llvm.ptr<array<4 x i8>>
  %0 = hw.bitcast %arg0 : (i32) -> !hw.array<4xi8>

// CHECK-NEXT: %[[AARG1:.*]] = builtin.unrealized_conversion_cast %arg1 : !hw.array<2xi32> to !llvm.array<2 x i32>
// CHECK-NEXT: %[[ONE2:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[A2:.*]] = llvm.alloca %[[ONE2]] x !llvm.array<2 x i32> {alignment = 4 : i64} : (i32) -> !llvm.ptr<array<2 x i32>>
// CHECK-NEXT: llvm.store %[[AARG1]], %[[A2]] : !llvm.ptr<array<2 x i32>>
// CHECK-NEXT: %[[B2:.*]] = llvm.bitcast %[[A2]] : !llvm.ptr<array<2 x i32>> to !llvm.ptr<i64>
// CHECK-NEXT: llvm.load %[[B2]] : !llvm.ptr<i64>
  %1 = hw.bitcast %arg1 : (!hw.array<2xi32>) -> i64

// CHECK-NEXT: %[[AARG2:.*]] = builtin.unrealized_conversion_cast %arg2 : !hw.struct<foo: i32, bar: i32> to !llvm.struct<(i32, i32)>
// CHECK-NEXT: %[[ONE3:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[A3:.*]] = llvm.alloca %[[ONE3]] x !llvm.struct<(i32, i32)> {alignment = 4 : i64} : (i32) -> !llvm.ptr<struct<(i32, i32)>>
// CHECK-NEXT: llvm.store %[[AARG2]], %[[A3]] : !llvm.ptr<struct<(i32, i32)>>
// CHECK-NEXT: %[[B3:.*]] = llvm.bitcast %[[A3]] : !llvm.ptr<struct<(i32, i32)>> to !llvm.ptr<i64>
// CHECK-NEXT: llvm.load %[[B3]] : !llvm.ptr<i64>
  %2 = hw.bitcast %arg2 : (!hw.struct<foo: i32, bar: i32>) -> i64

  return
}

// CHECK-LABEL: @convertArray
func.func @convertArray(%arg0 : i1, %arg1: !hw.array<2xi32>) {

  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: %[[ALLOCA:.*]] = llvm.alloca %[[ONE]] x !llvm.array<2 x i32> {alignment = 4 : i64} : (i32) -> !llvm.ptr<array<2 x i32>>
  // CHECK-NEXT: %[[CAST0:.*]] = builtin.unrealized_conversion_cast %arg1 : !hw.array<2xi32> to !llvm.array<2 x i32>
  // CHECK-NEXT: llvm.store %[[CAST0]], %[[ALLOCA]] : !llvm.ptr<array<2 x i32>>
  // CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-NEXT: %[[ZEXT:.*]] = llvm.zext %arg0 : i1 to i2
  // CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[ALLOCA]][%[[ZERO]], %[[ZEXT]]] : (!llvm.ptr<array<2 x i32>>, i32, i2) -> !llvm.ptr<i32>
  // CHECK-NEXT: llvm.load %[[GEP]] : !llvm.ptr<i32>
  %0 = hw.array_get %arg1[%arg0] : !hw.array<2xi32>, i1

  // CHECK-NEXT: %[[ZERO1:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-NEXT: %[[ONE4:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: %[[CAST1:.*]] = builtin.unrealized_conversion_cast %arg1 : !hw.array<2xi32> to !llvm.array<2 x i32>
  // CHECK-NEXT: %[[ALLOCA1:.*]] = llvm.alloca %[[ONE4]] x !llvm.array<2 x i32> {alignment = 4 : i64} : (i32) -> !llvm.ptr<array<2 x i32>>
  // CHECK-NEXT: llvm.store %[[CAST1]], %[[ALLOCA1]] : !llvm.ptr<array<2 x i32>>
  // CHECK-NEXT: %[[ZEXT1:.*]] = llvm.zext %arg0 : i1 to i2
  // CHECK-NEXT: %[[GEP1:.*]] = llvm.getelementptr %[[ALLOCA1]][%[[ZERO1]], %[[ZEXT1]]] : (!llvm.ptr<array<2 x i32>>, i32, i2) -> !llvm.ptr<i32>
  // CHECK-NEXT: %[[BITCAST:.*]] = llvm.bitcast %[[GEP1]] : !llvm.ptr<i32> to !llvm.ptr<array<1 x i32>>
  // CHECK-NEXT: llvm.load %[[LD:.*]] : !llvm.ptr<array<1 x i32>>
  %1 = hw.array_slice %arg1[%arg0] : (!hw.array<2xi32>) -> !hw.array<1xi32>

  // CHECK-NEXT: %[[UNDEF:.*]] = llvm.mlir.undef : !llvm.array<4 x i32>
  // CHECK-NEXT: %[[CAST2:.*]] = builtin.unrealized_conversion_cast %arg1 : !hw.array<2xi32> to !llvm.array<2 x i32>
  // CHECK-NEXT: %[[E1:.*]] = llvm.extractvalue %[[CAST2]][0] : !llvm.array<2 x i32>
  // CHECK-NEXT: %[[I1:.*]] = llvm.insertvalue %[[E1]], %[[UNDEF]][0] : !llvm.array<4 x i32>
  // CHECK-NEXT: %[[CAST3:.*]] = builtin.unrealized_conversion_cast %arg1 : !hw.array<2xi32> to !llvm.array<2 x i32>
  // CHECK-NEXT: %[[E2:.*]] = llvm.extractvalue %[[CAST3]][1] : !llvm.array<2 x i32>
  // CHECK-NEXT: %[[I2:.*]] = llvm.insertvalue %[[E2]], %[[I1]][1] : !llvm.array<4 x i32>
  // CHECK-NEXT: %[[CAST4:.*]] = builtin.unrealized_conversion_cast %arg1 : !hw.array<2xi32> to !llvm.array<2 x i32>
  // CHECK-NEXT: %[[E3:.*]] = llvm.extractvalue %[[CAST4]][0] : !llvm.array<2 x i32>
  // CHECK-NEXT: %[[I3:.*]] = llvm.insertvalue %[[E3]], %[[I2]][2] : !llvm.array<4 x i32>
  // CHECK-NEXT: %[[CAST5:.*]] = builtin.unrealized_conversion_cast %arg1 : !hw.array<2xi32> to !llvm.array<2 x i32>
  // CHECK: %[[E4:.*]] = llvm.extractvalue %[[CAST5]][1] : !llvm.array<2 x i32>
  // CHECK-NEXT: llvm.insertvalue %[[E4]], %[[I3]][3] : !llvm.array<4 x i32>
  %2 = hw.array_concat %arg1, %arg1 : !hw.array<2xi32>, !hw.array<2xi32>

  return
}

// CHECK-LABEL: @convertConstArray
func.func @convertConstArray(%arg0 : i1, %arg1: !hw.array<2xi32>) {
  // CHECK: %[[VAL_2:.*]] = llvm.mlir.addressof @_array_global : !llvm.ptr<array<2 x i32>>
  // CHECK-NEXT: %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<array<2 x i32>>
  // CHECK-NEXT: %[[VAL_4:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-NEXT: %[[VAL_5:.*]] = llvm.zext %arg0 : i1 to i2
  // CHECK-NEXT: %[[VAL_6:.*]] = llvm.getelementptr %[[VAL_2]][%[[VAL_4]], %[[VAL_5]]] : (!llvm.ptr<array<2 x i32>>, i32, i2) -> !llvm.ptr<i32>
  // CHECK-NEXT: %[[VAL_7:.*]] = llvm.load %[[VAL_6]] : !llvm.ptr<i32>
  %0 = hw.constant 0 : i32
  %1 = hw.constant 1 : i32
  %2 = hw.array_create %0, %1 : i32
  %3 = hw.array_get %2[%arg0] : !hw.array<2xi32>, i1
  return
}

// CHECK-LABEL: @convertStruct
func.func @convertStruct(%arg0 : i32, %arg1: !hw.struct<foo: i32, bar: i8>, %arg2: !hw.struct<>) {
  // COM: Produces 2 casts here - first one automatically, second one for use in extractvalue
  // CHECK-NEXT: %[[SCAST:.*]] = builtin.unrealized_conversion_cast %arg1 : !hw.struct<foo: i32, bar: i8> to !llvm.struct<(i8, i32)>
  // CHECK-NEXT: %[[SCAST0:.*]] = builtin.unrealized_conversion_cast %arg1 : !hw.struct<foo: i32, bar: i8> to !llvm.struct<(i8, i32)>
  // CHECK-NEXT: llvm.extractvalue %[[SCAST0]][1] : !llvm.struct<(i8, i32)>
  %0 = hw.struct_extract %arg1["foo"] : !hw.struct<foo: i32, bar: i8>

  // CHECK: %[[SCAST1:.*]] = builtin.unrealized_conversion_cast %arg1 : !hw.struct<foo: i32, bar: i8> to !llvm.struct<(i8, i32)>
  // CHECK: llvm.insertvalue %arg0, %[[SCAST1]][1] : !llvm.struct<(i8, i32)>
  %1 = hw.struct_inject %arg1["foo"], %arg0 : !hw.struct<foo: i32, bar: i8>

  // CHECK: llvm.extractvalue %[[SCAST]][1] : !llvm.struct<(i8, i32)>
  // CHECK: llvm.extractvalue %[[SCAST]][0] : !llvm.struct<(i8, i32)>
  %2:2 = hw.struct_explode %arg1 : !hw.struct<foo: i32, bar: i8>

  hw.struct_explode %arg2 : !hw.struct<>
  return
}
