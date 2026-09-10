// RUN: circt-opt %s --convert-hw-to-btor2="assume-init-reset" -o %t | FileCheck %s  

// CHECK: [[I32:[0-9]+]] sort bitvec 32
// CHECK: [[C0_I32:[0-9]+]] constd [[I32]] 0
// CHECK: [[REG1:[0-9]+]] state [[I32]] reg1
// CHECK: [[REG1_INIT:[0-9]+]] init [[I32]] [[REG1]] [[C0_I32]]
// CHECK: [[REG2:[0-9]+]] state [[I32]] reg2
// CHECK: [[REG2_INIT:[0-9]+]] init [[I32]] [[REG2]] [[C0_I32]]

module {
  hw.module @inc(in %a : i32, in %clk : !seq.clock, in %reset: i1) {
    // Basic case
    %c0_i32 = hw.constant 0 : i32
    %reg1 = seq.compreg %a, %clk reset %reset, %c0_i32 : i32
    // Check that reset value is used instead of initial value
    %initial_const = seq.initial () {
      %c42_i32 = hw.constant 42 : i32
      seq.yield %c42_i32 : i32
    } : () -> !seq.immutable<i32>
    %reg2 = seq.compreg %a, %clk reset %reset, %c0_i32 initial %initial_const : i32
  }
}
