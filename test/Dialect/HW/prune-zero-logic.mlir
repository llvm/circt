// RUN: circt-opt --hw-prune-zero-logic %s | FileCheck %s

// CHECK:      hw.module @test0() {
// CHECK-NEXT:   hw.output
// CHECK-NEXT: }

hw.module @test0(%arg : i0) -> (out: i0) {
    hw.output %arg : i0
}

// CHECK:      hw.module @test1(%d: i32, %clk: i1, %rst: i1) -> (out: i32) {
// CHECK-NEXT:   hw.output %d : i32
// CHECK-NEXT: }

hw.module @test1(%arg0 : i0, %d : i32, %arg1 : i0, %clk : i1, %rst : i1) -> (out: i32, out1: i0) {
    %0 = comb.sub %arg0, %arg1 : i0
    %1 = seq.compreg %1, %clk : i0
    hw.output %d, %1 : i32, i0
}
