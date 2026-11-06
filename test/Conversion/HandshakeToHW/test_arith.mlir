// RUN: circt-opt -lower-handshake-to-hw -split-input-file %s | FileCheck %s

// CHECK-LABEL: hw.module @arith_divui_in_ui32_ui32_out_ui32(
// CHECK:         comb.divu %{{.+}}, %{{.+}} : i32

handshake.func @test_divui(%arg0: i32, %arg1: i32, %arg2: none, ...) -> (i32, none) {
  %0 = arith.divui %arg0, %arg1 : i32
  return %0, %arg2 : i32, none
}

// -----

// CHECK-LABEL: hw.module @arith_divsi_in_ui32_ui32_out_ui32(
// CHECK:         comb.divs %{{.+}}, %{{.+}} : i32

handshake.func @test_divsi(%arg0: i32, %arg1: i32, %arg2: none, ...) -> (i32, none) {
  %0 = arith.divsi %arg0, %arg1 : i32
  return %0, %arg2 : i32, none
}
