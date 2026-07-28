// RUN: circt-opt %s --test-apply-lowering-options="options=emittedLineLength=9001,verifLabels" --export-verilog --verify-diagnostics | FileCheck %s

// CHECK-LABEL: module Labels
hw.module @Labels(in %a: i1) {
  // CHECK: foo1: assert property (a);
  // CHECK: foo2: assume property (a);
  // CHECK: foo3: cover property (a);
  sv.assert_property %a label "foo1" : i1
  sv.assume_property %a label "foo2" : i1
  sv.cover_property %a label "foo3" : i1

  // CHECK: bar: assert property (a);
  // CHECK: bar_0: assert property (a);
  sv.assert_property %a label "bar" : i1
  sv.assert_property %a label "bar" : i1
}

// CHECK-LABEL: module BasicEmissionNonTemporal
hw.module @BasicEmissionNonTemporal(in %a: i1, in %b: i1) {
  %0 = comb.and %a, %b : i1
  %1 = comb.or %a, %b : i1
  // CHECK: assert property (a);
  // CHECK: assume property (a & b);
  // CHECK: cover property (a | b);
  sv.assert_property %a : i1
  sv.assume_property %0 : i1
  sv.cover_property %1 : i1

  // CHECK: initial begin
  sv.initial {
    %2 = comb.xor %a, %b : i1
    %3 = comb.and %a, %b : i1
    // CHECK: assert(a);
    // CHECK: assume(a ^ b);
    // CHECK: cover(a & b);
    sv.assert_property %a : i1
    sv.assume_property %2 : i1
    sv.cover_property %3 : i1
  }
}

// CHECK-LABEL: module BasicEmissionTemporal
hw.module @BasicEmissionTemporal(in %a: i1) {
  %p = ltl.not %a : i1
  // CHECK: assert property (not a);
  // CHECK: assume property (not a);
  // CHECK: cover property (not a);
  sv.assert_property %p : !ltl.property
  sv.assume_property %p : !ltl.property
  sv.cover_property %p : !ltl.property

  %bc = ltl.boolean_constant true
  // CHECK: assert property (1'h1);
  sv.assert_property %bc : !ltl.property

  // CHECK: initial begin
  sv.initial {
    // CHECK: assert property (not a);
    // CHECK: assume property (not a);
    // CHECK: cover property (not a);
    sv.assert_property %p : !ltl.property
    sv.assume_property %p : !ltl.property
    sv.cover_property %p : !ltl.property
  }
}

// CHECK-LABEL: module Sequences
hw.module @Sequences(in %clk: i1, in %a: i1, in %b: i1) {
  // CHECK: assert property (@(posedge clk) ##0 a);
  %d0 = ltl.clocked_delay %a, posedge %clk, 0, 0 : i1
  sv.assert_property %d0 : !ltl.sequence
  // CHECK: assert property (@(posedge clk) ##[*] a);
  %d4 = ltl.clocked_delay %a, posedge %clk, 0 : i1
  sv.assert_property %d4 : !ltl.sequence
  // CHECK: assert property (@(posedge clk) ##[+] a);
  %d5 = ltl.clocked_delay %a, posedge %clk, 1 : i1
  sv.assert_property %d5 : !ltl.sequence

  // CHECK: assert property (@(posedge clk) ##4 a);
  %cd0 = ltl.clocked_delay %a, posedge %clk, 4, 0 : i1
  sv.assert_property %cd0 : !ltl.sequence
  // CHECK: assert property (@(negedge clk) ##[5:6] a);
  %cd1 = ltl.clocked_delay %a, negedge %clk, 5, 1 : i1
  sv.assert_property %cd1 : !ltl.sequence
  // CHECK: assert property (@(edge clk) ##[7:$] a);
  %cd2 = ltl.clocked_delay %a, edge %clk, 7 : i1
  sv.assert_property %cd2 : !ltl.sequence

  // CHECK: assert property (@(posedge clk) a);
  %ca = ltl.clocked_atom %a, posedge %clk : i1
  sv.assert_property %ca : !ltl.sequence

  // CHECK: assert property (@(posedge clk) a[*2:4]);
  %ra = ltl.clocked_atom %a, posedge %clk : i1
  %cr0 = ltl.clocked_repeat %ra, posedge %clk, 2, 2 : !ltl.sequence
  sv.assert_property %cr0 : !ltl.sequence
  // CHECK: assert property (@(posedge clk) a[->1:3]);
  %gra = ltl.clocked_atom %a, posedge %clk : i1
  %cgr0 = ltl.clocked_goto_repeat %gra, posedge %clk, 1, 2 : !ltl.sequence
  sv.assert_property %cgr0 : !ltl.sequence
  // CHECK: assert property (@(posedge clk) a[=1:3]);
  %nra = ltl.clocked_atom %a, posedge %clk : i1
  %cncr0 = ltl.clocked_non_consecutive_repeat %nra, posedge %clk, 1, 2 : !ltl.sequence
  sv.assert_property %cncr0 : !ltl.sequence

  // CHECK: assert property (a ##0 a);
  %c0 = ltl.concat %a, %a : i1, i1
  sv.assert_property %c0 : !ltl.sequence
  // CHECK: assert property (a ##0 (@(posedge clk) ##4 a));
  %c1 = ltl.concat %a, %cd0 : i1, !ltl.sequence
  sv.assert_property %c1 : !ltl.sequence
  // CHECK: assert property (a ##0 (@(posedge clk) ##4 a) ##0 (@(negedge clk) ##[5:6] a));
  %c2 = ltl.concat %a, %cd0, %cd1 : i1, !ltl.sequence, !ltl.sequence
  sv.assert_property %c2 : !ltl.sequence
  // CHECK: assert property ((@(posedge clk) ##4 a) ##0 (@(negedge clk) ##[5:6] a) ##0 (@(edge clk) ##[7:$] a));
  %c3 = ltl.concat %cd0, %cd1, %cd2 : !ltl.sequence, !ltl.sequence, !ltl.sequence
  sv.assert_property %c3 : !ltl.sequence
  // CHECK: assert property (a ##0 (@(posedge clk) ##4 b));
  %cd3 = ltl.clocked_delay %b, posedge %clk, 4, 0 : i1
  %c4 = ltl.concat %a, %cd3 : i1, !ltl.sequence
  sv.assert_property %c4 : !ltl.sequence

  // CHECK: assert property (a and (@(posedge clk) ##0 a));
  %g0 = ltl.and %a, %d0 : i1, !ltl.sequence
  sv.assert_property %g0 : !ltl.sequence
  // CHECK: assert property (a ##0 a and a ##0 (@(posedge clk) ##4 a));
  %g1 = ltl.and %c0, %c1 : !ltl.sequence, !ltl.sequence
  sv.assert_property %g1 : !ltl.sequence
  // CHECK: assert property (a or (@(posedge clk) ##0 a));
  %g2 = ltl.or %a, %d0 : i1, !ltl.sequence
  sv.assert_property %g2 : !ltl.sequence
  // CHECK: assert property (a ##0 a or a ##0 (@(posedge clk) ##4 a));
  %g3 = ltl.or %c0, %c1 : !ltl.sequence, !ltl.sequence
  sv.assert_property %g3 : !ltl.sequence

  // CHECK: assert property (@(posedge clk) a[*0]);
  %r0 = ltl.clocked_repeat %a, posedge %clk, 0, 0 : i1
  sv.assert_property %r0 : !ltl.sequence
  // CHECK: assert property (@(posedge clk) a[*4]);
  %r1 = ltl.clocked_repeat %a, posedge %clk, 4, 0 : i1
  sv.assert_property %r1 : !ltl.sequence
  // CHECK: assert property (@(posedge clk) a[*5:6]);
  %r2 = ltl.clocked_repeat %a, posedge %clk, 5, 1 : i1
  sv.assert_property %r2 : !ltl.sequence
  // CHECK: assert property (@(posedge clk) a[*7:$]);
  %r3 = ltl.clocked_repeat %a, posedge %clk, 7 : i1
  sv.assert_property %r3 : !ltl.sequence
  // CHECK: assert property (@(posedge clk) a[*]);
  %r4 = ltl.clocked_repeat %a, posedge %clk, 0 : i1
  sv.assert_property %r4 : !ltl.sequence
  // CHECK: assert property (@(posedge clk) a[+]);
  %r5 = ltl.clocked_repeat %a, posedge %clk, 1 : i1
  sv.assert_property %r5 : !ltl.sequence

  // CHECK: assert property (@(posedge clk) a[->0]);
  %gtr0 = ltl.clocked_goto_repeat %a, posedge %clk, 0, 0 : i1
  sv.assert_property %gtr0 : !ltl.sequence
  // CHECK: assert property (@(posedge clk) a[->4]);
  %gtr1 = ltl.clocked_goto_repeat %a, posedge %clk, 4, 0 : i1
  sv.assert_property %gtr1 : !ltl.sequence
  // CHECK: assert property (@(posedge clk) a[->5:6]);
  %gtr2 = ltl.clocked_goto_repeat %a, posedge %clk, 5, 1 : i1
  sv.assert_property %gtr2 : !ltl.sequence

  // CHECK: assert property (@(posedge clk) a[=0]);
  %ncr0 = ltl.clocked_non_consecutive_repeat %a, posedge %clk, 0, 0 : i1
  sv.assert_property %ncr0 : !ltl.sequence
  // CHECK: assert property (@(posedge clk) a[=4]);
  %ncr1 = ltl.clocked_non_consecutive_repeat %a, posedge %clk, 4, 0 : i1
  sv.assert_property %ncr1 : !ltl.sequence
  // CHECK: assert property (@(posedge clk) a[=5:6]);
  %ncr2 = ltl.clocked_non_consecutive_repeat %a, posedge %clk, 5, 1 : i1
  sv.assert_property %ncr2 : !ltl.sequence

  // CHECK: assert property (@(posedge clk) a);
  %k0 = ltl.clocked_atom %a, posedge %clk : i1
  sv.assert_property %k0 : !ltl.sequence
  // CHECK: assert property (@(negedge clk) a);
  %k1 = ltl.clocked_atom %a, negedge %clk : i1
  sv.assert_property %k1 : !ltl.sequence
  // CHECK: assert property (@(edge clk) a);
  %k2 = ltl.clocked_atom %a, edge %clk : i1
  sv.assert_property %k2 : !ltl.sequence
  // CHECK: assert property (@(posedge clk) ##4 a);
  %k3 = ltl.clocked_delay %a, posedge %clk, 4, 0 : i1
  sv.assert_property %k3 : !ltl.sequence
  // CHECK: assert property (b ##0 (@(posedge clk) a));
  %k4 = ltl.concat %b, %k0 : i1, !ltl.sequence
  sv.assert_property %k4 : !ltl.sequence
}

// CHECK-LABEL: module Properties
hw.module @Properties(in %clk: i1, in %a: i1, in %b: i1) {
  %true = hw.constant true

  // CHECK: assert property (not a);
  %n0 = ltl.not %a : i1
  sv.assert_property %n0 : !ltl.property

  // CHECK: assert property (a |-> b);
  // CHECK: assert property (a ##0 (@(posedge clk) ##1 b) |-> not a);
  // CHECK: assert property (a ##0 (@(posedge clk) ##1 b) ##0 (@(posedge clk) ##1 1'h1) |-> not a);
  %i0 = ltl.implication %a, %b : i1, i1
  sv.assert_property %i0 : !ltl.property
  %i1 = ltl.clocked_delay %b, posedge %clk, 1, 0 : i1
  %i2 = ltl.concat %a, %i1 : i1, !ltl.sequence
  %i3 = ltl.implication %i2, %n0 : !ltl.sequence, !ltl.property
  sv.assert_property %i3 : !ltl.property
  %i4 = ltl.clocked_delay %true, posedge %clk, 1, 0 : i1
  %i5 = ltl.concat %a, %i1, %i4 : i1, !ltl.sequence, !ltl.sequence
  %i6 = ltl.implication %i5, %n0 : !ltl.sequence, !ltl.property
  sv.assert_property %i6 : !ltl.property

  // CHECK: assert property (@(posedge clk) a until b);
  %u0 = ltl.clocked_until %a, posedge %clk, %b : i1, i1
  sv.assert_property %u0 : !ltl.property

  // CHECK: assert property (s_eventually a);
  %e0 = ltl.eventually %a : i1
  sv.assert_property %e0 : !ltl.property

  // Single-clock merge: one leading @(posedge clk), no inner events.
  // CHECK: assert property (@(posedge clk) a until b);
  %ua = ltl.clocked_atom %a, posedge %clk : i1
  %ub = ltl.clocked_atom %b, posedge %clk : i1
  %cu0 = ltl.clocked_until %ua, posedge %clk, %ub : !ltl.sequence, !ltl.sequence
  sv.assert_property %cu0 : !ltl.property

  // Eventually.
  // CHECK: assert property (@(posedge clk) s_eventually a);
  %ea = ltl.clocked_atom %a, posedge %clk : i1
  %cev0 = ltl.clocked_eventually %ea, posedge %clk : !ltl.sequence
  sv.assert_property %cev0 : !ltl.property

  // Emit `not(eventually(not(x)))` as `always x` (case A, cancelling nots).
  // CHECK: assert property (always a);
  %ag0 = ltl.not %a : i1
  %ag1 = ltl.eventually %ag0 : !ltl.property
  %ag2 = ltl.not %ag1 : !ltl.property
  sv.assert_property %ag2 : !ltl.property

  // Emit `not(eventually(x))` as `always not x` (case B, quantifier pull-up).
  // CHECK: assert property (always not a);
  %ag3 = ltl.eventually %a : i1
  %ag4 = ltl.not %ag3 : !ltl.property
  sv.assert_property %ag4 : !ltl.property

  // CHECK: assert property (@(posedge clk) disable iff (b) not a);
  sv.assert_property %n0 on posedge %clk disable_iff %b: !ltl.property
}

// CHECK-LABEL: module Precedence
hw.module @Precedence(in %clk: i1, in %a: i1, in %b: i1) {
  // CHECK: assert property ((a or (@(posedge clk) ##0 b)) and b);
  %a0 = ltl.clocked_delay %b, posedge %clk, 0, 0 : i1
  %a1 = ltl.or %a, %a0 : i1, !ltl.sequence
  %a2 = ltl.and %a1, %b : !ltl.sequence, i1
  sv.assert_property %a2 : !ltl.sequence

  // CHECK: assert property (@(posedge clk) ##1 (a or ##0 b));
  %d0 = ltl.clocked_delay %a1, posedge %clk, 1, 0 : !ltl.sequence
  sv.assert_property %d0 : !ltl.sequence

  // CHECK: assert property (not (a or (@(posedge clk) ##0 b)));
  %n0 = ltl.not %a1 : !ltl.sequence
  sv.assert_property %n0 : !ltl.property

  // CHECK: assert property (a and (a |-> b));
  %i0 = ltl.implication %a, %b : i1, i1
  %i1 = ltl.and %a, %i0 : i1, !ltl.property
  sv.assert_property %i1 : !ltl.property

  // CHECK: assert property ((s_eventually a) and b);
  // CHECK: assert property (b and (s_eventually a));
  %e0 = ltl.eventually %a : i1
  %e1 = ltl.and %e0, %b : !ltl.property, i1
  %e2 = ltl.and %b, %e0 : i1, !ltl.property
  sv.assert_property %e1 : !ltl.property
  sv.assert_property %e2 : !ltl.property

  // CHECK: assert property ((@(posedge clk) a until b) and a);
  %u0 = ltl.clocked_until %a, posedge %clk, %b : i1, i1
  %u1 = ltl.and %u0, %a : !ltl.property, i1
  sv.assert_property %u1 : !ltl.property

  // `always` sugar precedence. The inner value under `always` gets Qualifier
  // context (case A) or Unary context (case B, because it sits under the
  // synthesized `not`). `until` has Until precedence, which is tighter than
  // Qualifier but looser than Unary, so it differentiates the two.
  // CHECK: assert property (always (@(posedge clk) a until b));
  // CHECK: assert property (always not (@(posedge clk) a until b));
  %ag0 = ltl.not %u0 : !ltl.property
  %ag1 = ltl.eventually %ag0 : !ltl.property
  %ag2 = ltl.not %ag1 : !ltl.property
  sv.assert_property %ag2 : !ltl.property
  %ag3 = ltl.eventually %u0 : !ltl.property
  %ag4 = ltl.not %ag3 : !ltl.property
  sv.assert_property %ag4 : !ltl.property

  // `always ...` returns Qualifier precedence, so wrapping it in a tighter
  // context adds parens.
  // CHECK: assert property ((always a) and b);
  // CHECK: assert property (b and (always not a));
  %ag5 = ltl.not %a : i1
  %ag6 = ltl.eventually %ag5 : !ltl.property
  %ag7 = ltl.not %ag6 : !ltl.property
  %ag8 = ltl.and %ag7, %b : !ltl.property, i1
  sv.assert_property %ag8 : !ltl.property
  %ag9 = ltl.eventually %a : i1
  %ag10 = ltl.not %ag9 : !ltl.property
  %ag11 = ltl.and %b, %ag10 : i1, !ltl.property
  sv.assert_property %ag11 : !ltl.property
}

// CHECK-LABEL: module SystemVerilogSpecExamples
hw.module @SystemVerilogSpecExamples(in %clk: i1, in %a: i1, in %b: i1, in %c: i1, in %d: i1, in %e: i1) {
  // Section 16.7 "Sequences"

  // CHECK: assert property (a ##0 (@(posedge clk) ##1 b) ##0 c ##0 (@(posedge clk) ##1 d));
  %a0 = ltl.clocked_delay %b, posedge %clk, 1, 0 : i1
  %a1 = ltl.clocked_delay %d, posedge %clk, 1, 0 : i1
  %a2 = ltl.concat %a, %a0 : i1, !ltl.sequence
  %a3 = ltl.concat %c, %a1 : i1, !ltl.sequence
  %a4 = ltl.concat %a2, %a3 : !ltl.sequence, !ltl.sequence
  sv.assert_property %a4 : !ltl.sequence

  // Section 16.12.20 "Property examples"

  // CHECK: assert property ((@(posedge clk) ##1 a) |-> b);
  %d0 = ltl.clocked_delay %a, posedge %clk, 1, 0 : i1
  %d1 = ltl.implication %d0, %b : !ltl.sequence, i1
  sv.assert_property %d1 : !ltl.property
}

// CHECK-LABEL: module LivenessExample
hw.module @LivenessExample(in %clock: i1, in %reset: i1, in %isLive: i1) {
  %true = hw.constant true

  // CHECK: wire _GEN = ~isLive;
  // CHECK: assert property (disable iff (reset) (@(posedge clock) $fell(reset) & _GEN) |-> (@(posedge clock) s_eventually isLive));
  // CHECK: assume property (disable iff (reset) (@(posedge clock) $fell(reset) & _GEN) |-> (@(posedge clock) s_eventually isLive));
  %not_isLive = comb.xor %isLive, %true : i1
  %fell_reset = sv.verbatim.expr "$fell({{0}})"(%reset) : (i1) -> i1
  %0 = comb.and %fell_reset, %not_isLive : i1
  %clocked0 = ltl.clocked_atom %0, posedge %clock : i1
  %clockedLive = ltl.clocked_atom %isLive, posedge %clock : i1
  %1 = ltl.clocked_eventually %clockedLive, posedge %clock : !ltl.sequence
  %liveness_after_reset = ltl.implication %clocked0, %1 : !ltl.sequence, !ltl.property
  sv.assert_property %liveness_after_reset disable_iff %reset : !ltl.property
  sv.assume_property %liveness_after_reset disable_iff %reset : !ltl.property

  // CHECK: assert property (disable iff (reset) (@(posedge clock) isLive) ##0 (@(posedge clock) ##1 _GEN) |-> (@(posedge clock) s_eventually isLive));
  // CHECK-NEXT: assume property (disable iff (reset) (@(posedge clock) isLive) ##0 (@(posedge clock) ##1 _GEN) |-> (@(posedge clock) s_eventually isLive));
  %clockedNotLive = ltl.clocked_atom %not_isLive, posedge %clock : i1
  %4 = ltl.clocked_delay %clockedNotLive, posedge %clock, 1, 0 : !ltl.sequence
  %5 = ltl.concat %clockedLive, %4 : !ltl.sequence, !ltl.sequence
  %liveness_after_fall = ltl.implication %5, %1 : !ltl.sequence, !ltl.property
  sv.assert_property %liveness_after_fall disable_iff %reset : !ltl.property
  sv.assume_property %liveness_after_fall disable_iff %reset : !ltl.property
}

// https://github.com/llvm/circt/issues/5763
// CHECK-LABEL: module Issue5763
hw.module @Issue5763(in %a: i3) {
  // CHECK: assert property ((&a) & a[0]);
  %c-1_i3 = hw.constant -1 : i3
  %0 = comb.extract %a from 0 : (i3) -> i1
  %1 = comb.icmp bin eq %a, %c-1_i3 : i3
  %2 = comb.and bin %1, %0 : i1
  sv.assert_property %2 : i1
}

// CHECK-LABEL: module ClockedAsserts
hw.module @ClockedAsserts(in %clk: i1, in %a: i1, in %b: i1) {
  %true = hw.constant true
  %n0 = ltl.not %a : i1

  // CHECK: assert property (@(posedge clk) disable iff (b) not a);
  sv.assert_property %n0 on posedge %clk disable_iff %b : !ltl.property

  // CHECK: assume property (@(posedge clk) disable iff (b) not a);
  sv.assume_property %n0 on posedge %clk disable_iff %b : !ltl.property

  // CHECK: cover property (@(posedge clk) disable iff (b) not a);
  sv.cover_property %n0 on posedge %clk disable_iff %b: !ltl.property
}

// CHECK-LABEL: module Contracts
hw.module @Contracts(in %a: i42, out b : i42) {
  %0 = verif.contract %a : i42 {
  }
  // CHECK: assign b = a;
  hw.output %0 : i42
}

// CHECK-LABEL: module MultiClock
hw.module @MultiClock(in %clkA: i1, in %clkB: i1, in %a: i1, in %b: i1) {
  // Clockless combinators do not create an ambient clock. Same-clock sibling
  // branches therefore retain their individual events.
  // CHECK: assert property ((@(posedge clkA) a) and (@(posedge clkA) b));
  %sameA = ltl.clocked_atom %a, posedge %clkA : i1
  %sameB = ltl.clocked_atom %b, posedge %clkA : i1
  %same = ltl.and %sameA, %sameB : !ltl.sequence, !ltl.sequence
  sv.assert_property %same : !ltl.sequence

  // Two distinct clocks: the ambient-clock merge must not suppress either
  // event, so the inner @(...) stays inline.
  // CHECK: assert property (@(posedge clkA) a until (@(posedge clkB) b));
  %ca = ltl.clocked_atom %a, posedge %clkA : i1
  %cb = ltl.clocked_atom %b, posedge %clkB : i1
  %u  = ltl.clocked_until %ca, posedge %clkA, %cb : !ltl.sequence, !ltl.sequence
  sv.assert_property %u : !ltl.property
}
