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
  // CHECK: assert property (##0 a);
  %d0 = ltl.delay %a, 0, 0 : i1
  sv.assert_property %d0 : !ltl.sequence
  // CHECK: assert property (##4 a);
  %d1 = ltl.delay %a, 4, 0 : i1
  sv.assert_property %d1 : !ltl.sequence
  // CHECK: assert property (##[5:6] a);
  %d2 = ltl.delay %a, 5, 1 : i1
  sv.assert_property %d2 : !ltl.sequence
  // CHECK: assert property (##[7:$] a);
  %d3 = ltl.delay %a, 7 : i1
  sv.assert_property %d3 : !ltl.sequence
  // CHECK: assert property (##[*] a);
  %d4 = ltl.delay %a, 0 : i1
  sv.assert_property %d4 : !ltl.sequence
  // CHECK: assert property (##[+] a);
  %d5 = ltl.delay %a, 1 : i1
  sv.assert_property %d5 : !ltl.sequence

  // CHECK: assert property (a ##0 a);
  %c0 = ltl.concat %a, %a : i1, i1
  sv.assert_property %c0 : !ltl.sequence
  // CHECK: assert property (a ##4 a);
  %c1 = ltl.concat %a, %d1 : i1, !ltl.sequence
  sv.assert_property %c1 : !ltl.sequence
  // CHECK: assert property (a ##4 a ##[5:6] a);
  %c2 = ltl.concat %a, %d1, %d2 : i1, !ltl.sequence, !ltl.sequence
  sv.assert_property %c2 : !ltl.sequence
  // CHECK: assert property (##4 a ##[5:6] a ##[7:$] a);
  %c3 = ltl.concat %d1, %d2, %d3 : !ltl.sequence, !ltl.sequence, !ltl.sequence
  sv.assert_property %c3 : !ltl.sequence

  // CHECK: assert property (a and ##0 a);
  %g0 = ltl.and %a, %d0 : i1, !ltl.sequence
  sv.assert_property %g0 : !ltl.sequence
  // CHECK: assert property (a ##0 a and a ##4 a);
  %g1 = ltl.and %c0, %c1 : !ltl.sequence, !ltl.sequence
  sv.assert_property %g1 : !ltl.sequence
  // CHECK: assert property (a or ##0 a);
  %g2 = ltl.or %a, %d0 : i1, !ltl.sequence
  sv.assert_property %g2 : !ltl.sequence
  // CHECK: assert property (a ##0 a or a ##4 a);
  %g3 = ltl.or %c0, %c1 : !ltl.sequence, !ltl.sequence
  sv.assert_property %g3 : !ltl.sequence

  // CHECK: assert property (a[*0]);
  %r0 = ltl.repeat %a, 0, 0 : i1
  sv.assert_property %r0 : !ltl.sequence
  // CHECK: assert property (a[*4]);
  %r1 = ltl.repeat %a, 4, 0 : i1
  sv.assert_property %r1 : !ltl.sequence
  // CHECK: assert property (a[*5:6]);
  %r2 = ltl.repeat %a, 5, 1 : i1
  sv.assert_property %r2 : !ltl.sequence
  // CHECK: assert property (a[*7:$]);
  %r3 = ltl.repeat %a, 7 : i1
  sv.assert_property %r3 : !ltl.sequence
  // CHECK: assert property (a[*]);
  %r4 = ltl.repeat %a, 0 : i1
  sv.assert_property %r4 : !ltl.sequence
  // CHECK: assert property (a[+]);
  %r5 = ltl.repeat %a, 1 : i1
  sv.assert_property %r5 : !ltl.sequence

  // CHECK: assert property (a[->0]);
  %gtr0 = ltl.goto_repeat %a, 0, 0 : i1
  sv.assert_property %gtr0 : !ltl.sequence
  // CHECK: assert property (a[->4]);
  %gtr1 = ltl.goto_repeat %a, 4, 0 : i1
  sv.assert_property %gtr1 : !ltl.sequence
  // CHECK: assert property (a[->5:6]);
  %gtr2 = ltl.goto_repeat %a, 5, 1 : i1
  sv.assert_property %gtr2 : !ltl.sequence

  // CHECK: assert property (a[=0]);
  %ncr0 = ltl.non_consecutive_repeat %a, 0, 0 : i1
  sv.assert_property %ncr0 : !ltl.sequence
  // CHECK: assert property (a[=4]);
  %ncr1 = ltl.non_consecutive_repeat %a, 4, 0 : i1
  sv.assert_property %ncr1 : !ltl.sequence
  // CHECK: assert property (a[=5:6]);
  %ncr2 = ltl.non_consecutive_repeat %a, 5, 1 : i1
  sv.assert_property %ncr2 : !ltl.sequence

  // CHECK: assert property (@(posedge clk) a);
  %k0 = ltl.clock %a, posedge %clk : i1
  sv.assert_property %k0 : !ltl.sequence
  // CHECK: assert property (@(negedge clk) a);
  %k1 = ltl.clock %a, negedge %clk : i1
  sv.assert_property %k1 : !ltl.sequence
  // CHECK: assert property (@(edge clk) a);
  %k2 = ltl.clock %a, edge %clk : i1
  sv.assert_property %k2 : !ltl.sequence
  // CHECK: assert property (@(posedge clk) ##4 a);
  %k3 = ltl.clock %d1, posedge %clk : !ltl.sequence
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
  // CHECK: assert property (a ##1 b |-> not a);
  // CHECK: assert property (a ##1 b |=> not a);
  %i0 = ltl.implication %a, %b : i1, i1
  sv.assert_property %i0 : !ltl.property
  %i1 = ltl.delay %b, 1, 0 : i1
  %i2 = ltl.concat %a, %i1 : i1, !ltl.sequence
  %i3 = ltl.implication %i2, %n0 : !ltl.sequence, !ltl.property
  sv.assert_property %i3 : !ltl.property
  %i4 = ltl.delay %true, 1, 0 : i1
  %i5 = ltl.concat %a, %i1, %i4 : i1, !ltl.sequence, !ltl.sequence
  %i6 = ltl.implication %i5, %n0 : !ltl.sequence, !ltl.property
  sv.assert_property %i6 : !ltl.property

  // CHECK: assert property (a until b);
  %u0 = ltl.until %a, %b : i1, i1
  sv.assert_property %u0 : !ltl.property

  // CHECK: assert property (s_eventually a);
  %e0 = ltl.eventually %a : i1
  sv.assert_property %e0 : !ltl.property

  // CHECK: assert property (@(posedge clk) a |-> b);
  // CHECK: assert property (@(posedge clk) a ##1 b |-> (@(negedge b) not a));
  // CHECK: assert property (@(posedge clk) disable iff (b) not a);
  %k0 = ltl.clock %i0, posedge %clk : !ltl.property
  %k1 = ltl.clock %n0, negedge %b : !ltl.property
  %k2 = ltl.implication %i2, %k1 : !ltl.sequence, !ltl.property
  %k3 = ltl.clock %k2, posedge %clk : !ltl.property
  sv.assert_property %k0 : !ltl.property
  sv.assert_property %k3 : !ltl.property
  sv.assert_property %n0 on posedge %clk disable_iff %b: !ltl.property
}

// CHECK-LABEL: module Precedence
hw.module @Precedence(in %a: i1, in %b: i1) {
  // CHECK: assert property ((a or ##0 b) and b);
  %a0 = ltl.delay %b, 0, 0 : i1
  %a1 = ltl.or %a, %a0 : i1, !ltl.sequence
  %a2 = ltl.and %a1, %b : !ltl.sequence, i1
  sv.assert_property %a2 : !ltl.sequence

  // CHECK: assert property (##1 (a or ##0 b));
  %d0 = ltl.delay %a1, 1, 0 : !ltl.sequence
  sv.assert_property %d0 : !ltl.sequence

  // CHECK: assert property (not (a or ##0 b));
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

  // CHECK: assert property ((a until b) and a);
  %u0 = ltl.until %a, %b : i1, i1
  %u1 = ltl.and %u0, %a : !ltl.property, i1
  sv.assert_property %u1 : !ltl.property
}

// CHECK-LABEL: module SystemVerilogSpecExamples
hw.module @SystemVerilogSpecExamples(in %clk: i1, in %a: i1, in %b: i1, in %c: i1, in %d: i1, in %e: i1) {
  // Section 16.7 "Sequences"

  // CHECK: assert property (a ##1 b ##0 c ##1 d);
  %a0 = ltl.delay %b, 1, 0 : i1
  %a1 = ltl.delay %d, 1, 0 : i1
  %a2 = ltl.concat %a, %a0 : i1, !ltl.sequence
  %a3 = ltl.concat %c, %a1 : i1, !ltl.sequence
  %a4 = ltl.concat %a2, %a3 : !ltl.sequence, !ltl.sequence
  sv.assert_property %a4 : !ltl.sequence

  // Section 16.12.20 "Property examples"

  // CHECK: assert property (@(posedge clk) a |-> b ##1 c ##1 d);
  %b0 = ltl.delay %c, 1, 0 : i1
  %b1 = ltl.concat %b, %b0, %a1 : i1, !ltl.sequence, !ltl.sequence
  %b2 = ltl.implication %a, %b1 : i1, !ltl.sequence
  %b3 = ltl.clock %b2, posedge %clk : !ltl.property
  sv.assert_property %b3 : !ltl.property

  // CHECK: assert property (disable iff (e) @(posedge clk) a |-> not b ##1 c ##1 d);
  %c0 = ltl.not %b1 : !ltl.sequence
  %c1 = ltl.implication %a, %c0 : i1, !ltl.property
  %c3 = ltl.clock %c1, posedge %clk : !ltl.property
  sv.assert_property %c3 disable_iff %e : !ltl.property

  // CHECK: assert property (##1 a |-> b);
  %d0 = ltl.delay %a, 1, 0 : i1
  %d1 = ltl.implication %d0, %b : !ltl.sequence, i1
  sv.assert_property %d1 : !ltl.property
}

// CHECK-LABEL: module LivenessExample
hw.module @LivenessExample(in %clock: i1, in %reset: i1, in %isLive: i1) {
  %true = hw.constant true

  // CHECK: wire _GEN = ~isLive;
  // CHECK: assert property (disable iff (reset) @(posedge clock) $fell(reset) & _GEN |-> (s_eventually isLive));
  // CHECK: assume property (disable iff (reset) @(posedge clock) $fell(reset) & _GEN |-> (s_eventually isLive));
  %not_isLive = comb.xor %isLive, %true : i1
  %fell_reset = sv.verbatim.expr "$fell({{0}})"(%reset) : (i1) -> i1
  %0 = comb.and %fell_reset, %not_isLive : i1
  %1 = ltl.eventually %isLive : i1
  %2 = ltl.implication %0, %1 : i1, !ltl.property
  %liveness_after_reset = ltl.clock %2, posedge %clock : !ltl.property
  sv.assert_property %liveness_after_reset disable_iff %reset : !ltl.property
  sv.assume_property %liveness_after_reset disable_iff %reset : !ltl.property

  // CHECK: assert property (disable iff (reset) @(posedge clock) isLive ##1 _GEN |-> (s_eventually isLive));
  // CHECK-NEXT: assume property (disable iff (reset) @(posedge clock) isLive ##1 _GEN |-> (s_eventually isLive));
  %4 = ltl.delay %not_isLive, 1, 0 : i1
  %5 = ltl.concat %isLive, %4 : i1, !ltl.sequence
  %6 = ltl.implication %5, %1 : !ltl.sequence, !ltl.property
  %liveness_after_fall = ltl.clock %6, posedge %clock : !ltl.property
  sv.assert_property %liveness_after_fall disable_iff %reset : !ltl.property
  sv.assume_property %liveness_after_fall disable_iff %reset : !ltl.property
}

// https://github.com/llvm/circt/issues/5763
// CHECK-LABEL: module Issue5763
hw.module @Issue5763(in %a: i3) {
  // CHECK: assert property ((&a) & a[0]);
  %c-1_i3 = hw.constant -1 : i3
  %0 = comb.extract %a from 0 : (i3) -> i1
  %1 = comb.icmp eq %a, %c-1_i3 : i3
  %2 = comb.and %1, %0 : i1
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
