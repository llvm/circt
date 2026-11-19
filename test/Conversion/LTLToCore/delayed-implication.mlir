// RUN: circt-opt %s --lower-ltl-to-core | FileCheck %s

// Non-overlapping implication `a |=> b` as emitted by ImportVerilog:
// `concat(delay(a, 1, 0), true) |-> b`. The enable, `disable iff` in SVA, also
// cancels the obligation while it is in flight.
// CHECK-LABEL: hw.module @ImportVerilogForm
hw.module @ImportVerilogForm(in %a: i1, in %b: i1, in %clk: i1, in %rst: i1) {
  // CHECK: [[EN:%.+]] = comb.and %a, %rst : i1
  // CHECK: [[CLK:%.+]] = seq.to_clock %clk
  // CHECK: [[INIT:%.+]] = seq.initial
  // CHECK: [[REG:%.+]] = seq.compreg [[EN]], [[CLK]] initial [[INIT]] : i1
  // CHECK: [[NOT:%.+]] = comb.xor [[REG]], %{{.+}} : i1
  // CHECK: [[OR:%.+]] = comb.or [[NOT]], %b : i1
  // CHECK: verif.clocked_assert [[OR]] if %rst, posedge %clk : i1
  %true = hw.constant true
  %0 = ltl.delay %a, 1, 0 : i1
  %1 = ltl.concat %0, %true : !ltl.sequence, i1
  %2 = ltl.implication %1, %b : !ltl.sequence, i1
  verif.clocked_assert %2 if %rst, posedge %clk : !ltl.property
  hw.output
}

// The `ltl.concat`-documented spelling `concat(a, delay(true, 2, 0))`, with a
// two-cycle delay and an assume instead of an assert.
// CHECK-LABEL: hw.module @CanonicalForm
hw.module @CanonicalForm(in %a: i1, in %b: i1, in %clk: i1) {
  // CHECK: [[REG0:%.+]] = seq.compreg %a, %{{.+}} initial %{{.+}} : i1
  // CHECK: [[REG1:%.+]] = seq.compreg [[REG0]], %{{.+}} initial %{{.+}} : i1
  // CHECK: [[NOT:%.+]] = comb.xor [[REG1]], %{{.+}} : i1
  // CHECK: [[OR:%.+]] = comb.or [[NOT]], %b : i1
  // CHECK: verif.clocked_assume [[OR]], posedge %clk : i1
  %true = hw.constant true
  %0 = ltl.delay %true, 2, 0 : i1
  %1 = ltl.concat %a, %0 : i1, !ltl.sequence
  %2 = ltl.implication %1, %b : !ltl.sequence, i1
  verif.clocked_assume %2, posedge %clk : !ltl.property
  hw.output
}

// A negedge clock inverts the clock before sampling.
// CHECK-LABEL: hw.module @NegEdge
hw.module @NegEdge(in %a: i1, in %b: i1, in %clk: i1) {
  // CHECK: [[NOTCLK:%.+]] = comb.xor %clk, %{{.+}} : i1
  // CHECK: [[SEQCLK:%.+]] = seq.to_clock [[NOTCLK]]
  // CHECK: seq.compreg %a, [[SEQCLK]]
  // CHECK: verif.clocked_assert %{{.+}}, negedge %clk : i1
  %true = hw.constant true
  %0 = ltl.delay %a, 1, 0 : i1
  %1 = ltl.concat %0, %true : !ltl.sequence, i1
  %2 = ltl.implication %1, %b : !ltl.sequence, i1
  verif.clocked_assert %2, negedge %clk : !ltl.property
  hw.output
}

// Unsupported shapes are left untouched rather than erroring out.
// CHECK-LABEL: hw.module @Unsupported
hw.module @Unsupported(in %a: i1, in %b: i1, in %clk: i1, in %p: !ltl.property) {
  // An unbounded delay cannot be mapped to finite state.
  // CHECK: [[D0:%.+]] = ltl.delay %a, 1 :
  // CHECK: verif.clocked_assert
  %true = hw.constant true
  %0 = ltl.delay %a, 1 : i1
  %1 = ltl.concat %0, %true : !ltl.sequence, i1
  %2 = ltl.implication %1, %b : !ltl.sequence, i1
  verif.clocked_assert %2, posedge %clk : !ltl.property

  // A bounded window needs a token monitor, which this does not implement.
  // CHECK: ltl.delay %a, 1, 2 :
  // CHECK: verif.clocked_assert
  %3 = ltl.delay %a, 1, 2 : i1
  %4 = ltl.concat %3, %true : !ltl.sequence, i1
  %5 = ltl.implication %4, %b : !ltl.sequence, i1
  verif.clocked_assert %5, posedge %clk : !ltl.property

  // A non-boolean consequent is checked by an LTL-aware backend.
  // CHECK: ltl.implication
  // CHECK: verif.clocked_assert
  %6 = ltl.delay %a, 1, 0 : i1
  %7 = ltl.concat %6, %true : !ltl.sequence, i1
  %8 = ltl.implication %7, %p : !ltl.sequence, !ltl.property
  verif.clocked_assert %8, posedge %clk : !ltl.property

  // A dual-edge clocking event has no single sampling point.
  // CHECK: ltl.implication
  // CHECK: verif.clocked_assert
  %9 = ltl.delay %a, 1, 0 : i1
  %10 = ltl.concat %9, %true : !ltl.sequence, i1
  %11 = ltl.implication %10, %b : !ltl.sequence, i1
  verif.clocked_assert %11, edge %clk : !ltl.property
  hw.output
}

// `a |-> (b or ##1 c)`: if `a` holds without `b`, `c` is due in the next cycle.
// CHECK-LABEL: hw.module @OrNext
hw.module @OrNext(in %a: i1, in %b: i1, in %c: i1, in %clk: i1) {
  // CHECK: [[NOTB:%.+]] = comb.xor %b, %{{.+}} : i1
  // CHECK: [[PEND:%.+]] = comb.and %a, [[NOTB]] : i1
  // CHECK: [[REG:%.+]] = seq.compreg [[PEND]], %{{.+}} initial %{{.+}} : i1
  // CHECK: [[NOT:%.+]] = comb.xor [[REG]], %{{.+}} : i1
  // CHECK: [[OR:%.+]] = comb.or [[NOT]], %c : i1
  // CHECK: verif.clocked_assert [[OR]], posedge %clk : i1
  %0 = ltl.delay %c, 1, 0 : i1
  %1 = ltl.or %b, %0 : i1, !ltl.sequence
  %2 = ltl.implication %a, %1 : i1, !ltl.sequence
  verif.clocked_assert %2, posedge %clk : !ltl.property
  hw.output
}

// `a |-> ##[0:2] b`: an obligation stays pending for every cycle of the window
// in which `b` does not hold.
// CHECK-LABEL: hw.module @Window
hw.module @Window(in %a: i1, in %b: i1, in %clk: i1) {
  // CHECK: [[NOTB0:%.+]] = comb.xor %b, %{{.+}} : i1
  // CHECK: [[PEND0:%.+]] = comb.and %a, [[NOTB0]] : i1
  // CHECK: [[REG0:%.+]] = seq.compreg [[PEND0]], %{{.+}} initial %{{.+}} : i1
  // CHECK: [[NOTB1:%.+]] = comb.xor %b, %{{.+}} : i1
  // CHECK: [[PEND1:%.+]] = comb.and [[REG0]], [[NOTB1]] : i1
  // CHECK: [[REG1:%.+]] = seq.compreg [[PEND1]], %{{.+}} initial %{{.+}} : i1
  // CHECK: [[NOT:%.+]] = comb.xor [[REG1]], %{{.+}} : i1
  // CHECK: [[OR:%.+]] = comb.or [[NOT]], %b : i1
  // CHECK: verif.clocked_assume [[OR]], posedge %clk : i1
  %0 = ltl.delay %b, 0, 2 : i1
  %1 = ltl.implication %a, %0 : i1, !ltl.sequence
  verif.clocked_assume %1, posedge %clk : !ltl.property
  hw.output
}

// `disable iff (!en) a |=> ##[0:1] b`: every register samples the enable.
// CHECK-LABEL: hw.module @DelayedWindow
hw.module @DelayedWindow(in %a: i1, in %b: i1, in %clk: i1, in %en: i1) {
  // CHECK: [[EN0:%.+]] = comb.and %a, %en : i1
  // CHECK: [[REG0:%.+]] = seq.compreg [[EN0]], %{{.+}} initial %{{.+}} : i1
  // CHECK: [[NOTB:%.+]] = comb.xor %b, %{{.+}} : i1
  // CHECK: [[PEND:%.+]] = comb.and [[REG0]], [[NOTB]] : i1
  // CHECK: [[EN1:%.+]] = comb.and [[PEND]], %en : i1
  // CHECK: [[REG1:%.+]] = seq.compreg [[EN1]], %{{.+}} initial %{{.+}} : i1
  // CHECK: [[NOT:%.+]] = comb.xor [[REG1]], %{{.+}} : i1
  // CHECK: [[OR:%.+]] = comb.or [[NOT]], %b : i1
  // CHECK: verif.clocked_assert [[OR]] if %en, posedge %clk : i1
  %true = hw.constant true
  %0 = ltl.delay %a, 1, 0 : i1
  %1 = ltl.concat %0, %true : !ltl.sequence, i1
  %2 = ltl.delay %b, 0, 1 : i1
  %3 = ltl.implication %1, %2 : !ltl.sequence, !ltl.sequence
  verif.clocked_assert %3 if %en, posedge %clk : !ltl.property
  hw.output
}

// `##2 b`: `b` has to hold from the third cycle onwards.
// CHECK-LABEL: hw.module @BareDelay
hw.module @BareDelay(in %b: i1, in %clk: i1) {
  // CHECK: [[TRUE:%.+]] = hw.constant true
  // CHECK: [[REG0:%.+]] = seq.compreg [[TRUE]], %{{.+}} initial %{{.+}} : i1
  // CHECK: [[REG1:%.+]] = seq.compreg [[REG0]], %{{.+}} initial %{{.+}} : i1
  // CHECK: [[NOT:%.+]] = comb.xor [[REG1]], %{{.+}} : i1
  // CHECK: [[OR:%.+]] = comb.or [[NOT]], %b : i1
  // CHECK: verif.clocked_assert [[OR]], posedge %clk : i1
  %0 = ltl.delay %b, 2, 0 : i1
  verif.clocked_assert %0, posedge %clk : !ltl.sequence
  hw.output
}

