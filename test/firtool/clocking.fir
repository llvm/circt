; RUN: firtool %s | FileCheck %s

circuit Foo:
  intmodule ClockInverter:
    input in: Clock
    output out: Clock
    intrinsic = circt_clock_inv

  module Foo:
    input clk: Clock
    output inverted_clk: Clock

    inst inv of ClockInverter
    inv.in <= clk
    inverted_clk <= inv.out

  ; CHECK-LABEL: module Foo
  ; CHECK: assign inverted_clk = ~clk;
