// RUN: circt-opt %s | FileCheck %s --strict-whitespace

hw.module @foo(in %clk: !seq.clock, in %ce: i1, in %i: i32) {
  // CHECK: seq.compreg %{{[^,]*}}, %{{[^ ]*}} : i32
  seq.compreg %i, %clk : i32
  // CHECK: seq.compreg.ce %{{[^,]*}}, %{{[^,]*}}, %{{[^ ]*}} : i32
  seq.compreg.ce %i, %clk, %ce : i32
  // CHECK: seq.shiftreg[3] %{{[^,]*}}, %{{[^,]*}}, %{{[^ ]*}} : i32
  seq.shiftreg[3] %i, %clk, %ce : i32
}
