// RUN: circt-opt %s -allow-unregistered-dialect | FileCheck %s

// Test RefType with forceable=false (probe)
// CHECK-LABEL: func.func @test_ref_probe
func.func @test_ref_probe() {
  // CHECK: !hw.ref<i32, false>
  %0 = "test.dummy"() : () -> !hw.ref<i32, false>

  // CHECK: !hw.ref<i8, false>
  %1 = "test.dummy"() : () -> !hw.ref<i8, false>

  // CHECK: !hw.ref<i64, false>
  %2 = "test.dummy"() : () -> !hw.ref<i64, false>

  return
}

// Test RefType with forceable=true (rwprobe)
// CHECK-LABEL: func.func @test_ref_rwprobe
func.func @test_ref_rwprobe() {
  // CHECK: !hw.ref<i32, true>
  %0 = "test.dummy"() : () -> !hw.ref<i32, true>

  // CHECK: !hw.ref<i8, true>
  %1 = "test.dummy"() : () -> !hw.ref<i8, true>

  // CHECK: !hw.ref<!hw.array<4xi32>, true>
  %2 = "test.dummy"() : () -> !hw.ref<!hw.array<4xi32>, true>

  // CHECK: !hw.ref<!hw.struct<a: i32, b: i8>, true>
  %3 = "test.dummy"() : () -> !hw.ref<!hw.struct<a: i32, b: i8>, true>

  return
}

// Test various hardware types can be referenced
// CHECK-LABEL: func.func @test_ref_hw_types
func.func @test_ref_hw_types() {
  // Integer types
  // CHECK: !hw.ref<i1, false>
  %0 = "test.dummy"() : () -> !hw.ref<i1, false>

  // CHECK: !hw.ref<i32, false>
  %1 = "test.dummy"() : () -> !hw.ref<i32, false>

  // HW array types
  // CHECK: !hw.ref<!hw.array<8xi16>, false>
  %2 = "test.dummy"() : () -> !hw.ref<!hw.array<8xi16>, false>

  // HW struct types
  // CHECK: !hw.ref<!hw.struct<x: i32, y: i64, z: i8>, false>
  %3 = "test.dummy"() : () -> !hw.ref<!hw.struct<x: i32, y: i64, z: i8>, false>

  // Nested types
  // CHECK: !hw.ref<!hw.array<4xstruct<a: i8, b: i16>>, true>
  %4 = "test.dummy"() : () -> !hw.ref<!hw.array<4x!hw.struct<a: i8, b: i16>>, true>

  return
}
