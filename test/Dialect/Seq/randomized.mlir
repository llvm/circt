// RUN: circt-opt %s -verify-diagnostics --randomize-register-init | circt-opt -verify-diagnostics | FileCheck %s
hw.module @top(in %clk: !seq.clock, in %rst: i1, in %i: i42, in %s: !hw.struct<foo: i42>) {
  // CHECK: na
  %c0_i42 = hw.constant 0 : i42
  %r0 = seq.compreg %i, %clk reset %rst, %c0_i42 : i42
  seq.compreg %i, %clk : i42
}
