// RUN: circt-opt %s --split-input-file  --convert-hw-to-llvm=spill-arrays-early=false | FileCheck %s

// CHECK-LABEL: @convertArray
// CHECK-SAME: (%arg0: i1, %arg1: !llvm.array<2 x i32>,
func.func @convertArray(%arg0 : i1, %arg1: !hw.array<2xi32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
  // CHECK-NEXT: %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: %[[ALLOCA:.*]] = llvm.alloca %[[ONE]] x !llvm.array<2 x i32> {alignment = 4 : i64} : (i32) -> !llvm.ptr
  // CHECK-NEXT: llvm.store %arg1, %[[ALLOCA]] : !llvm.array<2 x i32>, !llvm.ptr
  // CHECK-NEXT: %[[ZEXT:.*]] = llvm.zext %arg0 : i1 to i2
  // CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[ALLOCA]][0, %[[ZEXT]]] : (!llvm.ptr, i2) -> !llvm.ptr, !llvm.array<2 x i32>
  // CHECK-NEXT: llvm.load %[[GEP]] : !llvm.ptr -> i32
  %0 = hw.array_get %arg1[%arg0] : !hw.array<2xi32>, i1

  // CHECK-NEXT: %[[ONE4:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: %[[ALLOCA1:.*]] = llvm.alloca %[[ONE4]] x !llvm.array<2 x i32> {alignment = 4 : i64} : (i32) -> !llvm.ptr
  // CHECK-NEXT: llvm.store %arg1, %[[ALLOCA1]] : !llvm.array<2 x i32>, !llvm.ptr
  // CHECK-NEXT: %[[ZEXT1:.*]] = llvm.zext %arg0 : i1 to i2
  // CHECK-NEXT: %[[GEP1:.*]] = llvm.getelementptr %[[ALLOCA1]][0, %[[ZEXT1]]] : (!llvm.ptr, i2) -> !llvm.ptr, !llvm.array<1 x i32>
  // CHECK-NEXT: llvm.load %[[GEP1]] : !llvm.ptr -> !llvm.array<1 x i32>
  %1 = hw.array_slice %arg1[%arg0] : (!hw.array<2xi32>) -> !hw.array<1xi32>

  // CHECK-NEXT: %[[UNDEF:.*]] = llvm.mlir.undef : !llvm.array<4 x i32>
  // CHECK-NEXT: %[[E1:.*]] = llvm.extractvalue %arg1[0] : !llvm.array<2 x i32>
  // CHECK-NEXT: %[[I1:.*]] = llvm.insertvalue %[[E1]], %[[UNDEF]][0] : !llvm.array<4 x i32>
  // CHECK-NEXT: %[[E2:.*]] = llvm.extractvalue %arg1[1] : !llvm.array<2 x i32>
  // CHECK-NEXT: %[[I2:.*]] = llvm.insertvalue %[[E2]], %[[I1]][1] : !llvm.array<4 x i32>
  // CHECK-NEXT: %[[E3:.*]] = llvm.extractvalue %arg1[0] : !llvm.array<2 x i32>
  // CHECK-NEXT: %[[I3:.*]] = llvm.insertvalue %[[E3]], %[[I2]][2] : !llvm.array<4 x i32>
  // CHECK: %[[E4:.*]] = llvm.extractvalue %arg1[1] : !llvm.array<2 x i32>
  // CHECK-NEXT: llvm.insertvalue %[[E4]], %[[I3]][3] : !llvm.array<4 x i32>
  %2 = hw.array_concat %arg1, %arg1 : !hw.array<2xi32>, !hw.array<2xi32>

  // CHECK-NEXT: [[V6:%.*]] = llvm.mlir.undef : !llvm.array<4 x i32>
  // CHECK-NEXT: [[V7:%.*]] = llvm.insertvalue %arg5, [[V6]][0] : !llvm.array<4 x i32>
  // CHECK-NEXT: [[V8:%.*]] = llvm.insertvalue %arg4, [[V7]][1] : !llvm.array<4 x i32>
  // CHECK-NEXT: [[V9:%.*]] = llvm.insertvalue %arg3, [[V8]][2] : !llvm.array<4 x i32>
  // CHECK-NEXT: [[V10:%.*]] = llvm.insertvalue %arg2, [[V9]][3] : !llvm.array<4 x i32>
  %3 = hw.array_create %arg2, %arg3, %arg4, %arg5 : i32

  return
}

// CHECK-LABEL: @convertArrayInject
func.func @convertArrayInject(
  %arg0: !hw.array<0xi32>, %arg1: !hw.array<1xi32>, %arg2: !hw.array<2xi32>, %arg3: !hw.array<5xi32>,
  %arg4: i32, %arg5: i64, %arg6: i1, %arg7: i3, %arg8: i64) {
  %0 = hw.array_inject %arg0[%arg5], %arg4 : !hw.array<0xi32>, i64

  // CHECK-NEXT: %[[ONE1:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: %[[ZEXT1:.*]] = llvm.zext %arg6 : i1 to i2
  // CHECK-NEXT: %[[MAX1:.*]] = llvm.mlir.constant(1 : i32) : i2
  // CHECK-NEXT: %[[UMIN1:.*]] = llvm.intr.umin(%[[ZEXT1]], %[[MAX1]]) : (i2, i2) -> i2
  // CHECK-NEXT: %[[ALLOCA1:.*]] = llvm.alloca %[[ONE1]] x !llvm.array<2 x i32> {alignment = 4 : i64} : (i32) -> !llvm.ptr
  // CHECK-NEXT: llvm.store %arg1, %[[ALLOCA1]] : !llvm.array<1 x i32>, !llvm.ptr
  // CHECK-NEXT: %[[GEP1:.*]] = llvm.getelementptr %[[ALLOCA1]][0, %[[UMIN1]]] : (!llvm.ptr, i2) -> !llvm.ptr, !llvm.array<2 x i32>
  // CHECK-NEXT: llvm.store %arg4, %[[GEP1]] : i32, !llvm.ptr
  // CHECK-NEXT: llvm.load %[[ALLOCA1]] : !llvm.ptr -> !llvm.array<1 x i32>
  %1 = hw.array_inject %arg1[%arg6], %arg4 : !hw.array<1xi32>, i1

  // CHECK-NEXT: %[[ONE2:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: %[[ZEXT2:.*]] = llvm.zext %arg6 : i1 to i2
  // CHECK-NEXT: %[[ALLOCA2:.*]] = llvm.alloca %[[ONE2]] x !llvm.array<2 x i32> {alignment = 4 : i64} : (i32) -> !llvm.ptr
  // CHECK-NEXT: llvm.store %arg2, %[[ALLOCA2]] : !llvm.array<2 x i32>, !llvm.ptr
  // CHECK-NEXT: %[[GEP2:.*]] = llvm.getelementptr %[[ALLOCA2]][0, %[[ZEXT2]]] : (!llvm.ptr, i2) -> !llvm.ptr, !llvm.array<2 x i32>
  // CHECK-NEXT: llvm.store %arg4, %[[GEP2]] : i32, !llvm.ptr
  // CHECK-NEXT: llvm.load %[[ALLOCA2]] : !llvm.ptr -> !llvm.array<2 x i32>
  %2 = hw.array_inject %arg2[%arg6], %arg4 : !hw.array<2xi32>, i1

  // CHECK-NEXT: %[[ONE3:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: %[[ZEXT3:.*]] = llvm.zext %arg7 : i3 to i4
  // CHECK-NEXT: %[[MAX3:.*]] = llvm.mlir.constant(5 : i32) : i4
  // CHECK-NEXT: %[[UMIN3:.*]] = llvm.intr.umin(%[[ZEXT3]], %[[MAX3]]) : (i4, i4) -> i4
  // CHECK-NEXT: %[[ALLOCA3:.*]] = llvm.alloca %[[ONE3]] x !llvm.array<6 x i32> {alignment = 4 : i64} : (i32) -> !llvm.ptr
  // CHECK-NEXT: llvm.store %arg3, %[[ALLOCA3]] : !llvm.array<5 x i32>, !llvm.ptr
  // CHECK-NEXT: %[[GEP3:.*]] = llvm.getelementptr %[[ALLOCA3]][0, %[[UMIN3]]] : (!llvm.ptr, i4) -> !llvm.ptr, !llvm.array<6 x i32>
  // CHECK-NEXT: llvm.store %arg4, %[[GEP3]] : i32, !llvm.ptr
  // CHECK-NEXT: llvm.load %[[ALLOCA3]] : !llvm.ptr -> !llvm.array<5 x i32>
  %3 = hw.array_inject %arg3[%arg7], %arg4 : !hw.array<5xi32>, i3

  // CHECK-NEXT: %[[ONE4:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: %[[ALLOCA4:.*]] = llvm.alloca %[[ONE4]] x !llvm.array<0 x i32> {alignment = 4 : i64} : (i32) -> !llvm.ptr
  // CHECK-NEXT: llvm.store %arg0, %[[ALLOCA4]] : !llvm.array<0 x i32>, !llvm.ptr
  // CHECK-NEXT: %[[ZEXT4:.*]] = llvm.zext %arg8 : i64 to i65
  // CHECK-NEXT: %[[GEP4:.*]] = llvm.getelementptr %[[ALLOCA4]][0, %[[ZEXT4]]] : (!llvm.ptr, i65) -> !llvm.ptr, !llvm.array<0 x i32>
  // CHECK-NEXT: llvm.load %[[GEP4]] : !llvm.ptr -> i32
  %4 = hw.array_get %0[%arg8] : !hw.array<0xi32>, i64

  return
}

// CHECK: llvm.mlir.global internal constant @[[GLOB1:.+]](dense<[1, 0]> : tensor<2xi32>) {addr_space = 0 : i32} : !llvm.array<2 x i32>

// CHECK: llvm.mlir.global internal constant @[[GLOB2:.+]](dense<{{[[][[]}}3, 2], [1, 0{{[]][]]}}> : tensor<2x2xi32>) {addr_space = 0 : i32} : !llvm.array<2 x array<2 x i32>>

// CHECK: llvm.mlir.global internal @[[GLOB3:.+]]() {addr_space = 0 : i32} : !llvm.array<2 x struct<(i1, i32)>> {
// CHECK-NEXT:   [[V0:%.+]] = llvm.mlir.undef : !llvm.array<2 x struct<(i1, i32)>>
// CHECK-NEXT:   [[V1:%.+]] = llvm.mlir.undef : !llvm.struct<(i1, i32)>
// CHECK-NEXT:   [[V2:%.+]] = llvm.mlir.constant(false) : i1
// CHECK-NEXT:   [[V3:%.+]] = llvm.insertvalue [[V2]], [[V1]][0] : !llvm.struct<(i1, i32)>
// CHECK-NEXT:   [[V4:%.+]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT:   [[V5:%.+]] = llvm.insertvalue [[V4]], [[V3]][1] : !llvm.struct<(i1, i32)>
// CHECK-NEXT:   [[V6:%.+]] = llvm.insertvalue [[V5]], [[V0]][0] : !llvm.array<2 x struct<(i1, i32)>>
// CHECK-NEXT:   [[V7:%.+]] = llvm.mlir.undef : !llvm.struct<(i1, i32)>
// CHECK-NEXT:   [[V8:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-NEXT:   [[V9:%.+]] = llvm.insertvalue [[V8]], [[V7]][0] : !llvm.struct<(i1, i32)>
// CHECK-NEXT:   [[V10:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:   [[V11:%.+]] = llvm.insertvalue [[V10]], [[V9]][1] : !llvm.struct<(i1, i32)>
// CHECK-NEXT:   [[V12:%.+]] = llvm.insertvalue [[V11]], [[V6]][1] : !llvm.array<2 x struct<(i1, i32)>>
// CHECK-NEXT:   llvm.return [[V12]] : !llvm.array<2 x struct<(i1, i32)>>
// CHECK-NEXT: }

// CHECK: @convertConstArray
func.func @convertConstArray(%arg0 : i1, %arg1 : i32) {
  // COM: Test: simple constant array converted to constant global
  // CHECK: %[[VAL_2:.*]] = llvm.mlir.addressof @[[GLOB1]] : !llvm.ptr
  // CHECK-NEXT: %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> !llvm.array<2 x i32>
  %0 = hw.aggregate_constant [0 : i32, 1 : i32] : !hw.array<2xi32>

  // COM: Test: nested constant array can also converted to a constant global
  // CHECK: %[[VAL_7:.*]] = llvm.mlir.addressof @[[GLOB2]] : !llvm.ptr
  // CHECK-NEXT: %{{.+}} = llvm.load %[[VAL_7]] : !llvm.ptr -> !llvm.array<2 x array<2 x i32>>
  %2 = hw.aggregate_constant [[0 : i32, 1 : i32], [2 : i32, 3 : i32]] : !hw.array<2x!hw.array<2xi32>>

  // COM: the same constants only create one global (note: even if they are in different functions).
  // CHECK: %[[VAL_8:.*]] = llvm.mlir.addressof @[[GLOB2]] : !llvm.ptr
  // CHECK-NEXT: %{{.+}} = llvm.load %[[VAL_8]] : !llvm.ptr -> !llvm.array<2 x array<2 x i32>>
  %3 = hw.aggregate_constant [[0 : i32, 1 : i32], [2 : i32, 3 : i32]] : !hw.array<2x!hw.array<2xi32>>

  // CHECK: %[[VAL_9:.+]] = llvm.mlir.addressof @[[GLOB3]] : !llvm.ptr
  // CHECK-NEXT: {{%.+}} = llvm.load %[[VAL_9]] : !llvm.ptr -> !llvm.array<2 x struct<(i1, i32)>>
  %4 = hw.aggregate_constant [[0 : i32, 1 : i1], [2 : i32, 0 : i1]] : !hw.array<2x!hw.struct<a: i32, b: i1>>

  // CHECK-NEXT: %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: %[[ZEXT0:.*]] = llvm.zext %arg0 : i1 to i2
  // CHECK-NEXT: %[[ALLOCA0:.*]] = llvm.alloca %[[ONE]] x !llvm.array<2 x i32> {alignment = 4 : i64} : (i32) -> !llvm.ptr
  // CHECK-NEXT: llvm.store %[[VAL_3]], %[[ALLOCA0]] : !llvm.array<2 x i32>, !llvm.ptr
  // CHECK-NEXT: %[[GEP0:.*]] = llvm.getelementptr %[[ALLOCA0]][0, %[[ZEXT0]]] : (!llvm.ptr, i2) -> !llvm.ptr, !llvm.array<2 x i32>
  // CHECK-NEXT: llvm.store %arg1, %[[GEP0]] : i32, !llvm.ptr
  // CHECK-NEXT: %{{.+}} = llvm.load %[[ALLOCA0]] : !llvm.ptr -> !llvm.array<2 x i32>
  %5 = hw.array_inject %0[%arg0], %arg1 : !hw.array<2xi32>, i1

  return
}

// CHECK: llvm.mlir.global internal @[[GLOB4:.+]]() {addr_space = 0 : i32} : !llvm.struct<(i2, i32)> {
// CHECK-NEXT:    [[V0:%.+]] = llvm.mlir.undef : !llvm.struct<(i2, i32)>
// CHECK-NEXT:    [[V1:%.+]] = llvm.mlir.constant(1 : i2) : i2
// CHECK-NEXT:    [[V2:%.+]] = llvm.insertvalue [[V1]], [[V0]][0] : !llvm.struct<(i2, i32)>
// CHECK-NEXT:    [[V3:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    [[V4:%.+]] = llvm.insertvalue [[V3]], [[V2]][1] : !llvm.struct<(i2, i32)>
// CHECK-NEXT:    llvm.return [[V4]] : !llvm.struct<(i2, i32)>
// CHECK-NEXT:  }

// CHECK: @convertConstStruct
func.func @convertConstStruct() {
  // CHECK-NEXT: [[V0:%.+]] = llvm.mlir.addressof @[[GLOB4]] : !llvm.ptr
  // CHECK-NEXT: [[V1:%.+]] = llvm.load [[V0]] : !llvm.ptr -> !llvm.struct<(i2, i32)>
  %0 = hw.aggregate_constant [0 : i32, 1 : i2] : !hw.struct<a: i32, b: i2>
  // CHECK-NEXT: {{%.+}} = llvm.extractvalue [[V1]][1] : !llvm.struct<(i2, i32)>
  %1 = hw.struct_extract %0["a"] : !hw.struct<a: i32, b: i2>
  return
}

// CHECK-LABEL: @convertStruct
// CHECK-SAME: (%arg0: i32, %arg1: !llvm.struct<(i8, i32)>, %arg2: !llvm.struct<()>, %arg3: i1, %arg4: i2)
func.func @convertStruct(%arg0 : i32, %arg1: !hw.struct<foo: i32, bar: i8>, %arg2: !hw.struct<>, %arg3 : i1, %arg4 : i2) {
  // CHECK-NEXT: llvm.extractvalue %arg1[1] : !llvm.struct<(i8, i32)>
  %0 = hw.struct_extract %arg1["foo"] : !hw.struct<foo: i32, bar: i8>

  // CHECK: llvm.insertvalue %arg0, %arg1[1] : !llvm.struct<(i8, i32)>
  %1 = hw.struct_inject %arg1["foo"], %arg0 : !hw.struct<foo: i32, bar: i8>

  // CHECK: llvm.extractvalue %arg1[1] : !llvm.struct<(i8, i32)>
  // CHECK: llvm.extractvalue %arg1[0] : !llvm.struct<(i8, i32)>
  %2:2 = hw.struct_explode %arg1 : !hw.struct<foo: i32, bar: i8>

  hw.struct_explode %arg2 : !hw.struct<>

  // CHECK-NEXT: [[V3:%.*]] = llvm.mlir.undef : !llvm.struct<(i32, i2, i1)>
  // CHECK-NEXT: [[V4:%.*]] = llvm.insertvalue %arg0, [[V3]][0] : !llvm.struct<(i32, i2, i1)>
  // CHECK-NEXT: [[V5:%.*]] = llvm.insertvalue %arg4, [[V4]][1] : !llvm.struct<(i32, i2, i1)>
  // CHECK-NEXT: [[V6:%.*]] = llvm.insertvalue %arg3, [[V5]][2] : !llvm.struct<(i32, i2, i1)>
  %3 = hw.struct_create (%arg3, %arg4, %arg0) : !hw.struct<foo: i1, bar: i2, baz: i32>

  return
}

// A union is represented as a flat byte buffer large enough to hold its widest
// member. Creating a union stores the member into a fresh buffer and reads the
// buffer back; extracting reverses the process.
// CHECK-LABEL: @convertUnion
// CHECK-SAME: (%arg0: i32, %arg1: i8, %[[ARG2:.*]]: !llvm.array<4 x i8>)
func.func @convertUnion(%arg0: i32, %arg1: i8, %arg2: !hw.union<foo: i32, bar: i8>) {
  // CHECK: %[[ONE0:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: %[[BUF0:.*]] = llvm.alloca %[[ONE0]] x !llvm.array<4 x i8> {alignment = 4 : i64} : (i32) -> !llvm.ptr
  // CHECK-NEXT: llvm.store %arg0, %[[BUF0]] : i32, !llvm.ptr
  // CHECK-NEXT: llvm.load %[[BUF0]] : !llvm.ptr -> !llvm.array<4 x i8>
  %0 = hw.union_create "foo", %arg0 : !hw.union<foo: i32, bar: i8>

  // CHECK: %[[ONE1:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: %[[BUF1:.*]] = llvm.alloca %[[ONE1]] x !llvm.array<4 x i8> {alignment = 1 : i64} : (i32) -> !llvm.ptr
  // CHECK-NEXT: llvm.store %arg1, %[[BUF1]] : i8, !llvm.ptr
  // CHECK-NEXT: llvm.load %[[BUF1]] : !llvm.ptr -> !llvm.array<4 x i8>
  %1 = hw.union_create "bar", %arg1 : !hw.union<foo: i32, bar: i8>

  // CHECK: %[[ONE2:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: %[[BUF2:.*]] = llvm.alloca %[[ONE2]] x !llvm.array<4 x i8> {alignment = 4 : i64} : (i32) -> !llvm.ptr
  // CHECK-NEXT: llvm.store %[[ARG2]], %[[BUF2]] : !llvm.array<4 x i8>, !llvm.ptr
  // CHECK-NEXT: llvm.load %[[BUF2]] : !llvm.ptr -> i32
  %2 = hw.union_extract %arg2["foo"] : !hw.union<foo: i32, bar: i8>

  // CHECK: %[[ONE3:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: %[[BUF3:.*]] = llvm.alloca %[[ONE3]] x !llvm.array<4 x i8> {alignment = 1 : i64} : (i32) -> !llvm.ptr
  // CHECK-NEXT: llvm.store %[[ARG2]], %[[BUF3]] : !llvm.array<4 x i8>, !llvm.ptr
  // CHECK-NEXT: llvm.load %[[BUF3]] : !llvm.ptr -> i8
  %3 = hw.union_extract %arg2["bar"] : !hw.union<foo: i32, bar: i8>

  return
}

// CHECK-LABEL: @nestedStructInject
// CHECK-SAME: (%arg0: !llvm.struct<(struct<(i1)>, i1)>, %arg1: !llvm.struct<(i1)>)
func.func @nestedStructInject(%arg0: !hw.struct<a: i1, b: !hw.struct<c: i1>>, %arg1: !hw.struct<c: i1>) {
  // CHECK-NEXT: llvm.insertvalue %arg1, %arg0[0] : !llvm.struct<(struct<(i1)>, i1)>
  %0 = hw.struct_inject %arg0["b"], %arg1 : !hw.struct<a: i1, b: !hw.struct<c: i1>>
  return
}

// CHECK-LABEL: @issue9171
func.func @issue9171(%idx: i1) -> i32 {
  // CHECK: [[ARRPTR:%.+]] = llvm.mlir.addressof
  // CHECK: [[ARRVAL:%.+]] = llvm.load [[ARRPTR]]
  %cst = hw.aggregate_constant [0 : i32, 1 : i32] : !hw.array<2xi32>

  // COM: Until we do proper RAW dependency checking array_get cannot assume
  // COM: the array's backing buffer to be immutable, so it has to rematerialize
  // COM: [[ARRVAL]] on the stack and must not reuse [[ARRPTR]].

  // CHECK: [[ALLOCA:%.+]] = llvm.alloca
  // CHECK: llvm.store [[ARRVAL]], [[ALLOCA]]
  // CHECK: [[VALPTR:%.+]] = llvm.getelementptr [[ALLOCA]]
  // CHECK: [[RETVAL:%.+]] = llvm.load [[VALPTR]]
  // CHECK: return [[RETVAL]]
  %get = hw.array_get %cst[%idx] : !hw.array<2xi32>, i1
  return %get : i32
}

// -----

// Check that the DLTI is considered when calculating the size for the union by over-aligning i64 to 16 byte.
module attributes {dlti.dl_spec = #dlti.dl_spec<i64 = dense<128> : vector<2xi64>>} {
  func.func @unionOfStructSize(%arg0: !hw.struct<foo: i64, bar: i64>, %arg1: !hw.union<s: !hw.struct<foo: i64, bar: i64>, i: i32>) {
    // CHECK: llvm.alloca %{{.*}} x !llvm.array<32 x i8>
    %0 = hw.union_create "s", %arg0 : !hw.union<s: !hw.struct<foo: i64, bar: i64>, i: i32>
    return
  }
}
