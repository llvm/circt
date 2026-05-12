// RUN: circt-opt %s -allow-unregistered-dialect | FileCheck %s

// Test RefType with forceable=false (probe)
// CHECK-LABEL: func.func @test_ref_probe
func.func @test_ref_probe() {
  // CHECK: !sim.ref<i32, false>
  %0 = "test.dummy"() : () -> !sim.ref<i32, false>

  // CHECK: !sim.ref<i8, false>
  %1 = "test.dummy"() : () -> !sim.ref<i8, false>

  // CHECK: !sim.ref<i64, false>
  %2 = "test.dummy"() : () -> !sim.ref<i64, false>

  return
}

// Test RefType with forceable=true (rwprobe)
// CHECK-LABEL: func.func @test_ref_rwprobe
func.func @test_ref_rwprobe() {
  // CHECK: !sim.ref<i32, true>
  %0 = "test.dummy"() : () -> !sim.ref<i32, true>

  // CHECK: !sim.ref<i8, true>
  %1 = "test.dummy"() : () -> !sim.ref<i8, true>

  // CHECK: !sim.ref<!hw.array<4xi32>, true>
  %2 = "test.dummy"() : () -> !sim.ref<!hw.array<4xi32>, true>

  // CHECK: !sim.ref<!hw.struct<a: i32, b: i8>, true>
  %3 = "test.dummy"() : () -> !sim.ref<!hw.struct<a: i32, b: i8>, true>

  return
}

// Test various hardware types can be referenced
// CHECK-LABEL: func.func @test_ref_hw_types
func.func @test_ref_hw_types() {
  // Integer types
  // CHECK: !sim.ref<i1, false>
  %0 = "test.dummy"() : () -> !sim.ref<i1, false>

  // CHECK: !sim.ref<i32, false>
  %1 = "test.dummy"() : () -> !sim.ref<i32, false>

  // HW array types
  // CHECK: !sim.ref<!hw.array<8xi16>, false>
  %2 = "test.dummy"() : () -> !sim.ref<!hw.array<8xi16>, false>

  // HW struct types
  // CHECK: !sim.ref<!hw.struct<x: i32, y: i64, z: i8>, false>
  %3 = "test.dummy"() : () -> !sim.ref<!hw.struct<x: i32, y: i64, z: i8>, false>

  // Nested types
  // CHECK: !sim.ref<!hw.array<4xstruct<a: i8, b: i16>>, true>
  %4 = "test.dummy"() : () -> !sim.ref<!hw.array<4x!hw.struct<a: i8, b: i16>>, true>

  return
}
