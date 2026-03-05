// RUN: circt-opt %s --convert-hw-to-btor2 -o %t | FileCheck %s  

// CHECK: [[I32:[0-9]+]] sort bitvec 32
// CHECK: [[A:[0-9]+]] input [[I32:[0-9]+]] a
// CHECK: [[REG:[0-9]+]] state [[I32]]
// CHECK: [[NEXT:[0-9]+]] next [[I32]] [[REG]] [[A]]  

module {
  hw.module @inc(in %a : i32, in %clk : i1) {
    %0 = seq.to_clock %clk
    %1 = seq.compreg %a, %0 : i32
  }
}
