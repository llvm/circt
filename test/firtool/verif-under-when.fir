; RUN: firtool %s | FileCheck %s
FIRRTL version 4.0.0

circuit Foo :
  public module Foo :
    input clock : Clock
    input a : UInt<1>
    input b : UInt<1>
    input c : UInt<1>
    input d : UInt<1>

    ; @(posedge clock) b |-> c
    node b_clocked = intrinsic(circt_ltl_clocked_atom<edge="posedge"> : UInt<1>, b, clock)
    node c_clocked = intrinsic(circt_ltl_clocked_atom<edge="posedge"> : UInt<1>, c, clock)
    node p1 = intrinsic(circt_ltl_implication : UInt<1>, b_clocked, c_clocked)

    when a :
      ; CHECK: assert property (disable iff (d)
      ; CHECK-NEXT: (@(posedge clock) a) and (@(posedge clock) b)
      ; CHECK-NEXT: |-> (@(posedge clock) c));
      ; CHECK: assume property (disable iff (d)
      ; CHECK-NEXT: (@(posedge clock) a) and (@(posedge clock) b)
      ; CHECK-NEXT: |-> (@(posedge clock) c));
      ; CHECK: cover property (disable iff (d) a and ((@(posedge clock) b) |-> (@(posedge clock) c)));
      intrinsic(circt_verif_assert, p1, not(d))
      intrinsic(circt_verif_assume, p1, not(d))
      intrinsic(circt_verif_cover, p1, not(d))

      ; CHECK: assert property ((@(posedge clock) a) and (@(posedge clock) b)
      ; CHECK-NEXT: |-> (@(posedge clock) c));
      ; CHECK: assume property ((@(posedge clock) a) and (@(posedge clock) b)
      ; CHECK-NEXT: |-> (@(posedge clock) c));
      ; CHECK: cover property (a and ((@(posedge clock) b) |-> (@(posedge clock) c)));
      intrinsic(circt_verif_assert, p1)
      intrinsic(circt_verif_assume, p1)
      intrinsic(circt_verif_cover, p1)

    else :
      ; CHECK: assert property (disable iff (d)
      ; CHECK-NEXT: (@(posedge clock) ~a) and (@(posedge clock) b)
      ; CHECK-NEXT: |-> (@(posedge clock) c));
      ; CHECK: assume property (disable iff (d)
      ; CHECK-NEXT: (@(posedge clock) ~a) and (@(posedge clock) b)
      ; CHECK-NEXT: |-> (@(posedge clock) c));
      ; CHECK: cover property (disable iff (d) ~a and ((@(posedge clock) b) |-> (@(posedge clock) c)));
      intrinsic(circt_verif_assert, p1, not(d))
      intrinsic(circt_verif_assume, p1, not(d))
      intrinsic(circt_verif_cover, p1, not(d))

      ; CHECK: assert property ((@(posedge clock) ~a) and (@(posedge clock) b)
      ; CHECK-NEXT: |-> (@(posedge clock) c));
      ; CHECK: assume property ((@(posedge clock) ~a) and (@(posedge clock) b)
      ; CHECK-NEXT: |-> (@(posedge clock) c));
      ; CHECK: cover property (~a and ((@(posedge clock) b) |-> (@(posedge clock) c)));
      intrinsic(circt_verif_assert, p1)
      intrinsic(circt_verif_assume, p1)
      intrinsic(circt_verif_cover, p1)

    ; CHECK: assert property (disable iff (d) (@(posedge clock) b) |-> (@(posedge clock) c));
    ; CHECK: assume property (disable iff (d) (@(posedge clock) b) |-> (@(posedge clock) c));
    ; CHECK: cover property (disable iff (d) (@(posedge clock) b) |-> (@(posedge clock) c));
    intrinsic(circt_verif_assert, p1, not(d))
    intrinsic(circt_verif_assume, p1, not(d))
    intrinsic(circt_verif_cover, p1, not(d))

    ; CHECK: assert property ((@(posedge clock) b) |-> (@(posedge clock) c));
    ; CHECK: assume property ((@(posedge clock) b) |-> (@(posedge clock) c));
    ; CHECK: cover property ((@(posedge clock) b) |-> (@(posedge clock) c));
    intrinsic(circt_verif_assert, p1)
    intrinsic(circt_verif_assume, p1)
    intrinsic(circt_verif_cover, p1)
