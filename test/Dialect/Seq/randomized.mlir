sv.macro.decl @FIRRTL_BEFORE_INITIAL
sv.macro.decl @FIRRTL_AFTER_INITIAL
sv.macro.decl @INIT_RANDOM_PROLOG_
// RUN: circt-opt %s -verify-diagnostics --randomize-register-init | circt-opt -verify-diagnostics | FileCheck %s
hw.module @top(in %clk: !seq.clock, in %rst: i1, in %i: i42, out o: i42, out j: i42) {
  // CHECK: na
  %c0_i42 = hw.constant 0 : i42
  %r0 = seq.compreg %i, %clk reset %rst, %c0_i42 : i42
  %r1 = seq.compreg %i, %clk : i42
  hw.output %r0, %r1: i42, i42
}
