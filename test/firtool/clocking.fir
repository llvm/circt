; RUN: firtool %s | FileCheck %s
FIRRTL version 4.0.0

circuit Foo:
  public module Foo:
    input clk: Clock
    output inverted_clk: Clock

    connect inverted_clk, intrinsic(circt_clock_inv : Clock, clk)

  ; CHECK-LABEL: module Foo
  ; CHECK: assign inverted_clk = ~clk;
