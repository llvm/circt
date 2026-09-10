; RUN: firtool %s | FileCheck %s
FIRRTL version 4.0.0

circuit Foo:
  public module Foo:
    input clk: Clock
    input a: UInt<1>
    input b: UInt<1>

    node delay = intrinsic(circt_ltl_delay<delay=1, length=0> : UInt<1>, b)
    node concat = intrinsic(circt_ltl_concat : UInt<1>, a, delay)
    node implication = intrinsic(circt_ltl_implication : UInt<1>, a, concat)
    node eventually = intrinsic(circt_ltl_eventually : UInt<1>, implication)

    ; CHECK: hello: assert property (s_eventually a |-> a ##1 b);
    intrinsic(circt_verif_assert<label="hello">, eventually)

