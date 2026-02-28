; RUN: firtool %s | FileCheck %s
FIRRTL version 4.0.0

circuit Foo :
  public module Foo :
    input clock : Clock
    input a : UInt<1>
    input b : UInt<1>
    input c : UInt<1>
    input d : UInt<1>

    ; b |-> c
    node p0 = intrinsic(circt_ltl_implication : UInt<1>, b, c)
    ; @(posedge clock) b |-> c
    node p1 = intrinsic(circt_ltl_clock : UInt<1>, p0, clock)

    when a :
      ; CHECK: assert property (@(posedge clock) disable iff (d) a & b |-> c);
      ; CHECK: assume property (@(posedge clock) disable iff (d) a & b |-> c);
      ; CHECK: cover property (@(posedge clock) disable iff (d) a and (b |-> c));
      intrinsic(circt_verif_assert, p1, not(d))
      intrinsic(circt_verif_assume, p1, not(d))
      intrinsic(circt_verif_cover, p1, not(d))

      ; CHECK: assert property (@(posedge clock) a & b |-> c);
      ; CHECK: assume property (@(posedge clock) a & b |-> c);
      ; CHECK: cover property (@(posedge clock) a and (b |-> c));
      intrinsic(circt_verif_assert, p1)
      intrinsic(circt_verif_assume, p1)
      intrinsic(circt_verif_cover, p1)

    else :
      ; CHECK: assert property (@(posedge clock) disable iff (d) ~a & b |-> c);
      ; CHECK: assume property (@(posedge clock) disable iff (d) ~a & b |-> c);
      ; CHECK: cover property (@(posedge clock) disable iff (d) ~a and (b |-> c));
      intrinsic(circt_verif_assert, p1, not(d))
      intrinsic(circt_verif_assume, p1, not(d))
      intrinsic(circt_verif_cover, p1, not(d))

      ; CHECK: assert property (@(posedge clock) ~a & b |-> c);
      ; CHECK: assume property (@(posedge clock) ~a & b |-> c);
      ; CHECK: cover property (@(posedge clock) ~a and (b |-> c));
      intrinsic(circt_verif_assert, p1)
      intrinsic(circt_verif_assume, p1)
      intrinsic(circt_verif_cover, p1)

    ; CHECK: assert property (@(posedge clock) disable iff (d) b |-> c);
    ; CHECK: assume property (@(posedge clock) disable iff (d) b |-> c);
    ; CHECK: cover property (@(posedge clock) disable iff (d) b |-> c);
    intrinsic(circt_verif_assert, p1, not(d))
    intrinsic(circt_verif_assume, p1, not(d))
    intrinsic(circt_verif_cover, p1, not(d))

    ; CHECK: assert property (@(posedge clock) b |-> c);
    ; CHECK: assume property (@(posedge clock) b |-> c);
    ; CHECK: cover property (@(posedge clock) b |-> c);
    intrinsic(circt_verif_assert, p1)
    intrinsic(circt_verif_assume, p1)
    intrinsic(circt_verif_cover, p1)
