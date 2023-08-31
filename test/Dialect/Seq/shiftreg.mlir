// RUN: circt-opt %s | circt-opt | FileCheck %s
// RUN: circt-opt %s --lower-seq-shiftreg | FileCheck %s --check-prefix=LO

// CHECK: %r0 = seq.shiftreg [3] %i, %clk, %ce, %rst, %c0_i32  : i32

// LO:    %r0_sh1 = seq.compreg.ce %i, %clk, %ce, %rst, %c0_i32  : i32
// LO:    %r0_sh2 = seq.compreg.ce %r0_sh1, %clk, %ce, %rst, %c0_i32  : i32
// LO:    %r0_sh3 = seq.compreg.ce %r0_sh2, %clk, %ce, %rst, %c0_i32  : i32
// LO:    hw.output %r0_sh3 : i32

hw.module @top(%clk: i1, %rst: i1, %ce: i1, %i: i32) -> (out : i32) {
  %rv = hw.constant 0 : i32
  %r0 = seq.shiftreg [3] %i, %clk, %ce, %rst, %rv : i32
  hw.output %r0 : i32
}
