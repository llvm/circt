// RUN: circt-opt --verify-clocked-assert-like %s --split-input-file --verify-diagnostics | circt-opt

hw.module @verifyDisables(in %clk: i1, in %a: i1, in %b: i1) {
  %n0 = ltl.not %a : i1

  // expected-error @below {{Nested clock or disable operations are not allowed for clock_assertlike operations.}}
  %disabled0 = ltl.disable %n0 if %b : !ltl.property
  verif.clocked_assert %disabled0 clock posedge %clk disable %b : !ltl.property
}

// -----

hw.module @verifyDisables1(in %clk: i1, in %a: i1, in %b: i1) {
  %n0 = ltl.not %a : i1

  // expected-error @below {{Nested clock or disable operations are not allowed for clock_assertlike operations.}}
  %disabled1 = ltl.disable %n0 if %b : !ltl.property
  verif.clocked_assume %disabled1 clock posedge %clk disable %b : !ltl.property
}

// -----

hw.module @verifyDisables2(in %clk: i1, in %a: i1, in %b: i1) {
  %n0 = ltl.not %a : i1

  // expected-error @below {{Nested clock or disable operations are not allowed for clock_assertlike operations.}}
  %disabled2 = ltl.disable %n0 if %b : !ltl.property
  verif.clocked_cover %disabled2 clock posedge %clk disable %b : !ltl.property
}

// -----

hw.module @verifyClocks(in %clk: i1, in %a: i1, in %b: i1) {
  %n0 = ltl.not %a : i1

  // expected-error @below {{Nested clock or disable operations are not allowed for clock_assertlike operations.}}
  %clocked = ltl.clock %n0, posedge %clk : !ltl.property
  verif.clocked_assert %clocked  clock posedge %clk disable %b: !ltl.property
}

// -----

hw.module @verifyClocks1(in %clk: i1, in %a: i1, in %b: i1) {
  %n0 = ltl.not %a : i1

  // expected-error @below {{Nested clock or disable operations are not allowed for clock_assertlike operations.}}
  %clocked = ltl.clock %n0, posedge %clk : !ltl.property
  verif.clocked_assume %clocked clock posedge %clk disable %b: !ltl.property
}

// -----

hw.module @verifyClocks2(in %clk: i1, in %a: i1, in %b: i1) {
  %n0 = ltl.not %a : i1

  // expected-error @below {{Nested clock or disable operations are not allowed for clock_assertlike operations.}}
  %clocked = ltl.clock %n0, posedge %clk : !ltl.property
  verif.clocked_cover %clocked clock posedge %clk disable %b: !ltl.property
}

// -----

hw.module @deeplynested(in %clk: i1, in %a: i1, in %b: i1) {
  %n0 = ltl.not %a : i1
  // expected-error @below {{Nested clock or disable operations are not allowed for clock_assertlike operations.}}
  %clocked = ltl.clock %n0, posedge %clk : !ltl.property

  %e1 = ltl.eventually %clocked : !ltl.property
  %i1 = ltl.implication %b, %e1 : i1, !ltl.property
  %a1 = ltl.and %b, %i1 : i1, !ltl.property
  %o1 = ltl.or %b, %a1 : i1, !ltl.property

  verif.clocked_assert %o1 clock posedge %clk disable %b : !ltl.property
}