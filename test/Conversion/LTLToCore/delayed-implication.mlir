// RUN: circt-opt %s --lower-ltl-to-core | FileCheck %s

// `b` must hold one cycle after `a`, with an enable on the assert.
// CHECK-LABEL: hw.module @SingleDelayAssert
hw.module @SingleDelayAssert(in %a: i1, in %b: i1, in %clk: i1, in %rst: i1) {
  // CHECK: [[EN:%.+]] = comb.and %a, %rst : i1
  // CHECK: [[CLK:%.+]] = seq.to_clock %clk
  // CHECK: [[INIT:%.+]] = seq.initial() {
  // CHECK-NEXT: [[FALSE:%.+]] = hw.constant false
  // CHECK-NEXT: seq.yield [[FALSE]] : i1
  // CHECK-NEXT: } : () -> !seq.immutable<i1>
  // CHECK: [[REG:%.+]] = seq.compreg [[EN]], [[CLK]] initial [[INIT]] : i1
  // CHECK: [[TRUE:%.+]] = hw.constant true
  // CHECK-NEXT: [[NOT:%.+]] = comb.xor [[REG]], [[TRUE]] : i1
  // CHECK: [[OR:%.+]] = comb.or [[NOT]], %b : i1
  // CHECK: verif.clocked_assert [[OR]] if %rst, posedge %clk : i1
  %true = hw.constant true
  %0 = ltl.delay %true, 1, 0 : i1
  %1 = ltl.concat %a, %0 : i1, !ltl.sequence
  %2 = ltl.implication %1, %b : !ltl.sequence, i1
  verif.clocked_assert %2 if %rst, posedge %clk : !ltl.property
  hw.output
}

// `b` must hold two cycles after `a`.
// CHECK-LABEL: hw.module @TwoCycleDelayAssume
hw.module @TwoCycleDelayAssume(in %a: i1, in %b: i1, in %clk: i1) {
  // CHECK: [[REG0:%.+]] = seq.compreg %a, %{{.+}} initial %{{.+}} : i1
  // CHECK: [[REG1:%.+]] = seq.compreg [[REG0]], %{{.+}} initial %{{.+}} : i1
  // CHECK: [[TRUE:%.+]] = hw.constant true
  // CHECK-NEXT: [[NOT:%.+]] = comb.xor [[REG1]], [[TRUE]] : i1
  // CHECK: [[OR:%.+]] = comb.or [[NOT]], %b : i1
  // CHECK: verif.clocked_assume [[OR]], posedge %clk : i1
  %true = hw.constant true
  %0 = ltl.delay %true, 2, 0 : i1
  %1 = ltl.concat %a, %0 : i1, !ltl.sequence
  %2 = ltl.implication %1, %b : !ltl.sequence, i1
  verif.clocked_assume %2, posedge %clk : !ltl.property
  hw.output
}

// CHECK-LABEL: hw.module @NegEdge
hw.module @NegEdge(in %a: i1, in %b: i1, in %clk: i1) {
  // CHECK: hw.constant true
  // CHECK: [[TRUE:%.+]] = hw.constant true
  // CHECK-NEXT: [[NOTCLK:%.+]] = comb.xor %clk, [[TRUE]] : i1
  // CHECK: [[SEQCLK:%.+]] = seq.to_clock [[NOTCLK]]
  // CHECK: seq.compreg %a, [[SEQCLK]]
  // CHECK: verif.clocked_assert %{{.+}}, negedge %clk : i1
  %true = hw.constant true
  %0 = ltl.delay %true, 1, 0 : i1
  %1 = ltl.concat %a, %0 : i1, !ltl.sequence
  %2 = ltl.implication %1, %b : !ltl.sequence, i1
  verif.clocked_assert %2, negedge %clk : !ltl.property
  hw.output
}

// Unsupported shapes are left untouched rather than erroring out. The checks
// cover the whole module, so they also make sure no registers or comb ops are
// created for them.
// CHECK-LABEL: hw.module @Unsupported
hw.module @Unsupported(in %a: i1, in %b: i1, in %clk: i1, in %p: !ltl.property) {
  // Antecedent with a delay before `a`.
  // CHECK-NEXT: [[D0:%.+]] = ltl.delay %a, 1, 0 : i1
  // CHECK-NEXT: [[C0:%.+]] = ltl.concat [[D0]], [[TRUE:%.+]] : !ltl.sequence, i1
  // CHECK-NEXT: [[I0:%.+]] = ltl.implication [[C0]], %b : !ltl.sequence, i1
  // CHECK-NEXT: verif.clocked_assert [[I0]], posedge %clk : !ltl.property
  %12 = ltl.delay %a, 1, 0 : i1
  %13 = ltl.concat %12, %true : !ltl.sequence, i1
  %14 = ltl.implication %13, %b : !ltl.sequence, i1
  verif.clocked_assert %14, posedge %clk : !ltl.property

  // Unbounded delay in the antecedent.
  // CHECK-NEXT: [[TRUE]] = hw.constant true
  // CHECK-NEXT: [[D1:%.+]] = ltl.delay %a, 1 : i1
  // CHECK-NEXT: [[C1:%.+]] = ltl.concat [[D1]], [[TRUE]] : !ltl.sequence, i1
  // CHECK-NEXT: [[I1:%.+]] = ltl.implication [[C1]], %b : !ltl.sequence, i1
  // CHECK-NEXT: verif.clocked_assert [[I1]], posedge %clk : !ltl.property
  %true = hw.constant true
  %0 = ltl.delay %a, 1 : i1
  %1 = ltl.concat %0, %true : !ltl.sequence, i1
  %2 = ltl.implication %1, %b : !ltl.sequence, i1
  verif.clocked_assert %2, posedge %clk : !ltl.property

  // Bounded window in the antecedent.
  // CHECK-NEXT: [[D2:%.+]] = ltl.delay %a, 1, 2 : i1
  // CHECK-NEXT: [[C2:%.+]] = ltl.concat [[D2]], [[TRUE]] : !ltl.sequence, i1
  // CHECK-NEXT: [[I2:%.+]] = ltl.implication [[C2]], %b : !ltl.sequence, i1
  // CHECK-NEXT: verif.clocked_assert [[I2]], posedge %clk : !ltl.property
  %3 = ltl.delay %a, 1, 2 : i1
  %4 = ltl.concat %3, %true : !ltl.sequence, i1
  %5 = ltl.implication %4, %b : !ltl.sequence, i1
  verif.clocked_assert %5, posedge %clk : !ltl.property

  // Non-boolean consequent.
  // CHECK-NEXT: [[D3:%.+]] = ltl.delay [[TRUE]], 1, 0 : i1
  // CHECK-NEXT: [[C3:%.+]] = ltl.concat %a, [[D3]] : i1, !ltl.sequence
  // CHECK-NEXT: [[I3:%.+]] = ltl.implication [[C3]], %p : !ltl.sequence, !ltl.property
  // CHECK-NEXT: verif.clocked_assert [[I3]], posedge %clk : !ltl.property
  %6 = ltl.delay %true, 1, 0 : i1
  %7 = ltl.concat %a, %6 : i1, !ltl.sequence
  %8 = ltl.implication %7, %p : !ltl.sequence, !ltl.property
  verif.clocked_assert %8, posedge %clk : !ltl.property

  // Dual-edge clocking.
  // CHECK-NEXT: [[D4:%.+]] = ltl.delay [[TRUE]], 1, 0 : i1
  // CHECK-NEXT: [[C4:%.+]] = ltl.concat %a, [[D4]] : i1, !ltl.sequence
  // CHECK-NEXT: [[I4:%.+]] = ltl.implication [[C4]], %b : !ltl.sequence, i1
  // CHECK-NEXT: verif.clocked_assert [[I4]], edge %clk : !ltl.property
  %9 = ltl.delay %true, 1, 0 : i1
  %10 = ltl.concat %a, %9 : i1, !ltl.sequence
  %11 = ltl.implication %10, %b : !ltl.sequence, i1
  verif.clocked_assert %11, edge %clk : !ltl.property

  // CHECK-NEXT: hw.output
  hw.output
}

// `a |-> (b, delay(c, 1, 0))`
// If `a` holds without `b`, `c` is due in the next cycle.
// CHECK-LABEL: hw.module @OrNext
hw.module @OrNext(in %a: i1, in %b: i1, in %c: i1, in %clk: i1) {
  // CHECK: [[TRUE:%.+]] = hw.constant true
  // CHECK-NEXT: [[NOTB:%.+]] = comb.xor %b, [[TRUE]] : i1
  // CHECK: [[PEND:%.+]] = comb.and %a, [[NOTB]] : i1
  // CHECK: [[REG:%.+]] = seq.compreg [[PEND]], %{{.+}} initial %{{.+}} : i1
  // CHECK: [[TRUE:%.+]] = hw.constant true
  // CHECK-NEXT: [[NOT:%.+]] = comb.xor [[REG]], [[TRUE]] : i1
  // CHECK: [[OR:%.+]] = comb.or [[NOT]], %c : i1
  // CHECK: verif.clocked_assert [[OR]], posedge %clk : i1
  %0 = ltl.delay %c, 1, 0 : i1
  %1 = ltl.or %b, %0 : i1, !ltl.sequence
  %2 = ltl.implication %a, %1 : i1, !ltl.sequence
  verif.clocked_assert %2, posedge %clk : !ltl.property
  hw.output
}

// `a |-> delay(b, 0, 2)`
// An obligation stays pending for every cycle of the window in which `b` does
// not hold.
// CHECK-LABEL: hw.module @Window
hw.module @Window(in %a: i1, in %b: i1, in %clk: i1) {
  // CHECK: [[TRUE:%.+]] = hw.constant true
  // CHECK-NEXT: [[NOTB0:%.+]] = comb.xor %b, [[TRUE]] : i1
  // CHECK: [[PEND0:%.+]] = comb.and %a, [[NOTB0]] : i1
  // CHECK: [[REG0:%.+]] = seq.compreg [[PEND0]], %{{.+}} initial %{{.+}} : i1
  // CHECK: [[TRUE:%.+]] = hw.constant true
  // CHECK-NEXT: [[NOTB1:%.+]] = comb.xor %b, [[TRUE]] : i1
  // CHECK: [[PEND1:%.+]] = comb.and [[REG0]], [[NOTB1]] : i1
  // CHECK: [[REG1:%.+]] = seq.compreg [[PEND1]], %{{.+}} initial %{{.+}} : i1
  // CHECK: [[TRUE:%.+]] = hw.constant true
  // CHECK-NEXT: [[NOT:%.+]] = comb.xor [[REG1]], [[TRUE]] : i1
  // CHECK: [[OR:%.+]] = comb.or [[NOT]], %b : i1
  // CHECK: verif.clocked_assume [[OR]], posedge %clk : i1
  %0 = ltl.delay %b, 0, 2 : i1
  %1 = ltl.implication %a, %0 : i1, !ltl.sequence
  verif.clocked_assume %1, posedge %clk : !ltl.property
  hw.output
}

// CHECK-LABEL: hw.module @DelayedWindow
hw.module @DelayedWindow(in %a: i1, in %b: i1, in %clk: i1, in %en: i1) {
  // CHECK: [[EN0:%.+]] = comb.and %a, %en : i1
  // CHECK: [[REG0:%.+]] = seq.compreg [[EN0]], %{{.+}} initial %{{.+}} : i1
  // CHECK: [[TRUE:%.+]] = hw.constant true
  // CHECK-NEXT: [[NOTB:%.+]] = comb.xor %b, [[TRUE]] : i1
  // CHECK: [[PEND:%.+]] = comb.and [[REG0]], [[NOTB]] : i1
  // CHECK: [[EN1:%.+]] = comb.and [[PEND]], %en : i1
  // CHECK: [[REG1:%.+]] = seq.compreg [[EN1]], %{{.+}} initial %{{.+}} : i1
  // CHECK: [[TRUE:%.+]] = hw.constant true
  // CHECK-NEXT: [[NOT:%.+]] = comb.xor [[REG1]], [[TRUE]] : i1
  // CHECK: [[OR:%.+]] = comb.or [[NOT]], %b : i1
  // CHECK: verif.clocked_assert [[OR]] if %en, posedge %clk : i1
  %true = hw.constant true
  %0 = ltl.delay %true, 1, 0 : i1
  %1 = ltl.concat %a, %0 : i1, !ltl.sequence
  %2 = ltl.delay %b, 0, 1 : i1
  %3 = ltl.implication %1, %2 : !ltl.sequence, !ltl.sequence
  verif.clocked_assert %3 if %en, posedge %clk : !ltl.property
  hw.output
}

