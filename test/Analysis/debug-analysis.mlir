// RUN: circt-opt %s --test-debug-analysis | FileCheck %s

// CHECK-LABEL: @Foo(
hw.module @Foo(out z: i42) {
  // CHECK: hw.constant 0 : i42 {debug.only}
  // CHECK: dbg.variable "a", {{%.+}} {debug.only}
  %c0_i42 = hw.constant 0 : i42
  dbg.variable "a", %c0_i42 : i42

  // CHECK: hw.constant 1 : i42
  // CHECK-NOT: debug.only
  // CHECK: dbg.variable "b", {{%.+}} {debug.only}
  %c1_i42 = hw.constant 1 : i42
  dbg.variable "b", %c1_i42 : i42

  hw.output %c1_i42 : i42
}

// CHECK-LABEL: @Empty(
// CHECK-NOT: debug.only
hw.module @Empty() {}

// CHECK-LABEL: @DebugOnlyBody(
hw.module @DebugOnlyBody(in %a: i1, in %b: i1) {
  // CHECK: comb.and {{.+}} {debug.only}
  // CHECK: dbg.struct {{.+}} {debug.only}
  // CHECK: dbg.variable "a", {{%.+}} {debug.only}
  // CHECK: dbg.variable "b", {{%.+}} {debug.only}
  // CHECK: dbg.variable "c", {{%.+}} {debug.only}
  %0 = comb.and %a, %b : i1
  %1 = dbg.struct {"x": %0} : i1
  dbg.variable "a", %a : i1
  dbg.variable "b", %b : i1
  dbg.variable "c", %1 : !dbg.struct
}
