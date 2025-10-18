// RUN: circt-opt --verify-clocked-assert-like %s --split-input-file --verify-diagnostics | circt-opt

// -----

hw.module @verifyClocks(in %clk: !seq.clock, in %a: i1, in %b: i1) {
  %n0 = ltl.not %a : i1

  // expected-error @below {{Nested clock or disable operations are not allowed for clock_assertlike operations.}}
  %clocked = ltl.clock %n0, posedge %clk : !ltl.property
  verif.clocked_assert %clocked if %b, posedge %clk : !ltl.property
}

// -----

hw.module @verifyClocks1(in %clk: !seq.clock, in %a: i1, in %b: i1) {
  %n0 = ltl.not %a : i1

  // expected-error @below {{Nested clock or disable operations are not allowed for clock_assertlike operations.}}
  %clocked = ltl.clock %n0, posedge %clk : !ltl.property
  verif.clocked_assume %clocked if %b, posedge %clk : !ltl.property
}

// -----

hw.module @verifyClocks2(in %clk: !seq.clock, in %a: i1, in %b: i1) {
  %n0 = ltl.not %a : i1

  // expected-error @below {{Nested clock or disable operations are not allowed for clock_assertlike operations.}}
  %clocked = ltl.clock %n0, posedge %clk : !ltl.property
  verif.clocked_cover %clocked if %b, posedge %clk : !ltl.property
}

// -----

hw.module @deeplynested(in %clk: !seq.clock, in %a: i1, in %b: i1) {
  %n0 = ltl.not %a : i1
  // expected-error @below {{Nested clock or disable operations are not allowed for clock_assertlike operations.}}
  %clocked = ltl.clock %n0, posedge %clk : !ltl.property

  %e1 = ltl.eventually %clocked : !ltl.property
  %i1 = ltl.implication %b, %e1 : i1, !ltl.property
  %a1 = ltl.and %b, %i1 : i1, !ltl.property
  %o1 = ltl.or %b, %a1 : i1, !ltl.property

  verif.clocked_assert %o1 if %b, posedge %clk : !ltl.property
}

// -----

hw.module @clockedarg(in %clocked: !ltl.property, in %a: i1, in %clk: !seq.clock) {
  verif.clocked_assert %clocked if %a, posedge %clk : !ltl.property
}
