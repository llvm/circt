; RUN: firtool %s | FileCheck %s
FIRRTL version 4.0.0

circuit Foo:
  public module Foo:
    input clock: Clock
    input a: UInt<1>
    input b: UInt<1>
    input c: UInt<1>

    ; `a until b`, built entirely from explicit-clock LTL intrinsics and
    ; asserted via a plain `circt_verif_assert` (no `circt_ltl_clock`, no
    ; `verif.clocked_assert` involved). The ambient-clock merge performed at
    ; ExportVerilog emission time collapses the two clocked leaves down to a
    ; single leading `@(posedge clock)`.
    node ca = intrinsic(circt_ltl_clocked_atom<edge="posedge"> : UInt<1>, a, clock)
    node cb = intrinsic(circt_ltl_clocked_atom<edge="posedge"> : UInt<1>, b, clock)
    node u  = intrinsic(circt_ltl_clocked_until<edge="posedge"> : UInt<1>, ca, clock, cb)
    ; CHECK: assert property (@(posedge clock) a until b);
    intrinsic(circt_verif_assert, u)

    ; `eventually c`, again fully explicit-clock and merged to one clock.
    node cc = intrinsic(circt_ltl_clocked_atom<edge="posedge"> : UInt<1>, c, clock)
    node ev = intrinsic(circt_ltl_clocked_eventually<edge="posedge"> : UInt<1>, cc, clock)
    ; CHECK: assert property (@(posedge clock) s_eventually c);
    intrinsic(circt_verif_assert, ev)

    ; `a[*2:5]`, exercising the clocked_repeat family too.
    node rp = intrinsic(circt_ltl_clocked_repeat<base=2, more=3, edge="posedge"> : UInt<1>, ca, clock)
    ; CHECK: assert property (@(posedge clock) a[*2:5]);
    intrinsic(circt_verif_assert, rp)
