; RUN: firtool %s | FileCheck %s
FIRRTL version 4.0.0

circuit Foo:
  public module Foo:
    input clock: Clock
    input a: UInt<1>
    input b: UInt<1>

    node delay = intrinsic(circt_ltl_clocked_delay<delay=2, length=0, edge="negedge"> : UInt<1>, b, clock)
    node prop = intrinsic(circt_ltl_implication : UInt<1>, a, delay)
    node clocked = intrinsic(circt_ltl_clock : UInt<1>, prop, clock)

    ; CHECK: assert property (@(posedge clock) a |-> (@(negedge clock) ##2 b));
    intrinsic(circt_verif_assert, clocked)
