// RUN: circt-opt --canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @Casts
func.func @Casts(%arg0: !moore.i1) -> (!moore.i1, !moore.i1) {
  // CHECK-NOT: moore.conversion
  // CHECK-NOT: moore.bool_cast
  %0 = moore.conversion %arg0 : !moore.i1 -> !moore.i1
  %1 = moore.bool_cast %arg0 : !moore.i1 -> !moore.i1
  // CHECK: return %arg0, %arg0
  return %0, %1 : !moore.i1, !moore.i1
}

// CHECK-LABEL: moore.module @SingleAssign
moore.module @SingleAssign() {
  // CHECK-NOT: moore.variable
  // CHECK: %a = moore.assigned_variable %0 : <i32>
  %a = moore.variable : <i32>
  // CHECK: %0 = moore.constant 32 : i32
  %0 = moore.constant 32 : i32
  // CHECK: moore.assign %a, %0 : i32
  moore.assign %a, %0 : i32
  moore.output
}

// CHECK-LABEL: moore.module @MultiAssign
moore.module @MultiAssign() {
  // CHECK-NOT: moore.assigned_variable
  // CHECK: %a = moore.variable : <i32>
  %a = moore.variable : <i32>
  // CHECK: %0 = moore.constant 32 : i32
  %0 = moore.constant 32 : i32
  // CHECK: moore.assign %a, %0 : i32
  moore.assign %a, %0 : i32
  // CHECK: %1 = moore.constant 64 : i32
  %1 = moore.constant 64 : i32
  // CHECK: moore.assign %a, %1 : i32
  moore.assign %a, %1 : i32
  moore.output
}

// CHECK-LABEL: func.func @StructExtractFold1
func.func @StructExtractFold1(%arg0: !moore.struct<{a: i17, b: i42}>, %arg1: !moore.i17) -> (!moore.i17) {
  // CHECK-NEXT: return %arg1 : !moore.i17
  %0 = moore.struct_inject %arg0, "a", %arg1 : struct<{a: i17, b: i42}>, i17
  %1 = moore.struct_extract %0, "a" : struct<{a: i17, b: i42}> -> i17
  return %1 : !moore.i17
}

// CHECK-LABEL: func.func @StructExtractFold2
func.func @StructExtractFold2(%arg0: !moore.i17, %arg1: !moore.i42) -> (!moore.i17, !moore.i42) {
  // CHECK-NEXT: return %arg0, %arg1 : !moore.i17, !moore.i42
  %0 = moore.struct_create %arg0, %arg1 : !moore.i17, !moore.i42 -> struct<{a: i17, b: i42}>
  %1 = moore.struct_extract %0, "a" : struct<{a: i17, b: i42}> -> i17
  %2 = moore.struct_extract %0, "b" : struct<{a: i17, b: i42}> -> i42
  return %1, %2 : !moore.i17, !moore.i42
}

// CHECK-LABEL: func.func @StructInjectFold1
func.func @StructInjectFold1(%arg0: !moore.struct<{a: i32, b: i32}>) -> (!moore.struct<{a: i32, b: i32}>) {
  // CHECK-NEXT: [[C42:%.+]] = moore.constant 42
  // CHECK-NEXT: [[C43:%.+]] = moore.constant 43
  // CHECK-NEXT: [[TMP:%.+]] = moore.struct_create [[C42]], [[C43]] : !moore.i32, !moore.i32 -> struct<{a: i32, b: i32}>
  // CHECK-NEXT: return [[TMP]]
  %0 = moore.constant 42 : i32
  %1 = moore.constant 43 : i32
  %2 = moore.struct_inject %arg0, "a", %1 : struct<{a: i32, b: i32}>, i32
  %3 = moore.struct_inject %2, "b", %1 : struct<{a: i32, b: i32}>, i32
  %4 = moore.struct_inject %3, "a", %0 : struct<{a: i32, b: i32}>, i32
  return %4 : !moore.struct<{a: i32, b: i32}>
}

// CHECK-LABEL: func.func @StructInjectFold2
func.func @StructInjectFold2() -> (!moore.struct<{a: i32, b: i32}>) {
  // CHECK-NEXT: [[C42:%.+]] = moore.constant 42
  // CHECK-NEXT: [[C43:%.+]] = moore.constant 43
  // CHECK-NEXT: [[TMP:%.+]] = moore.struct_create [[C42]], [[C43]] : !moore.i32, !moore.i32 -> struct<{a: i32, b: i32}>
  // CHECK-NEXT: return [[TMP]]
  %0 = moore.constant 42 : i32
  %1 = moore.constant 43 : i32
  %2 = moore.struct_create %0, %0 : !moore.i32, !moore.i32 -> struct<{a: i32, b: i32}>
  %3 = moore.struct_inject %2, "b", %1 : struct<{a: i32, b: i32}>, i32
  return %3 : !moore.struct<{a: i32, b: i32}>
}

// CHECK-LABEL: func.func @StructInjectFold3
func.func @StructInjectFold3(%arg0: !moore.struct<{a: i32, b: i32}>) -> (!moore.struct<{a: i32, b: i32}>) {
  // CHECK-NEXT: [[C43:%.+]] = moore.constant 43
  // CHECK-NEXT: [[TMP:%.+]] = moore.struct_inject %arg0, "a", [[C43]] : struct<{a: i32, b: i32}>, i32
  // CHECK-NEXT: return [[TMP]]
  %0 = moore.constant 42 : i32
  %1 = moore.constant 43 : i32
  %2 = moore.struct_inject %arg0, "a", %0 : struct<{a: i32, b: i32}>, i32
  %3 = moore.struct_inject %2, "a", %1 : struct<{a: i32, b: i32}>, i32
  return %3 : !moore.struct<{a: i32, b: i32}>
}
