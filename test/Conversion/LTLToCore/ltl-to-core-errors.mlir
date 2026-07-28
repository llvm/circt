// RUN: circt-opt %s --lower-ltl-to-core --split-input-file --verify-diagnostics

hw.module @UnboundedClockedDelay(in %clock : !seq.clock, in %a : i1) {
  %0 = seq.from_clock %clock
  %1 = ltl.clocked_atom %a, posedge %0 : i1
  // expected-error @below {{lower-ltl-to-core only supports bounded LTL delays}}
  %2 = ltl.clocked_delay %1, posedge %0, 1 : !ltl.sequence
  verif.assert %2 : !ltl.sequence
  hw.output
}

// -----

hw.module @UnclockedDelay(in %clock : !seq.clock, in %a : i1) {
  %0 = seq.from_clock %clock
  %1 = ltl.clocked_atom %a, posedge %0 : i1
  // expected-error @below {{lower-ltl-to-core does not support temporal LTL operation 'ltl.delay'}}
  %2 = ltl.delay %1, 1, 0 : !ltl.sequence
  verif.assert %2 : !ltl.sequence
  hw.output
}

// -----

hw.module @ClockScope(in %clock : !seq.clock, in %a : i1) {
  %0 = seq.from_clock %clock
  // expected-error @below {{lower-ltl-to-core does not support temporal LTL operation 'ltl.clock'}}
  %1 = ltl.clock %a, posedge %0 : i1
  verif.assert %1 : !ltl.sequence
  hw.output
}

// -----

hw.module @BothEdgeClock(in %clock : !seq.clock, in %a : i1) {
  %0 = seq.from_clock %clock
  // expected-error @below {{lower-ltl-to-core does not support both-edge LTL clocks}}
  %1 = ltl.clocked_atom %a, edge %0 : i1
  verif.assert %1 : !ltl.sequence
  hw.output
}

// -----

hw.module @DifferentEdges(in %clock : i1, in %a : i1, in %b : i1) {
  %0 = ltl.clocked_atom %a, posedge %clock : i1
  %1 = ltl.clocked_atom %b, negedge %clock : i1
  %2 = ltl.clocked_delay %0, posedge %clock, 1, 0 : !ltl.sequence
  %3 = ltl.clocked_delay %1, negedge %clock, 1, 0 : !ltl.sequence
  // expected-error @below {{lower-ltl-to-core cannot align temporal LTL timing paths}}
  %4 = ltl.and %2, %3 : !ltl.sequence, !ltl.sequence
  verif.assert %4 : !ltl.sequence
  hw.output
}

// -----

hw.module @UnsupportedTemporalOp(in %clock : !seq.clock, in %a : i1, in %b : i1) {
  %0 = seq.from_clock %clock
  %1 = ltl.clocked_atom %a, posedge %0 : i1
  %2 = ltl.clocked_atom %b, posedge %0 : i1
  // expected-error @below {{lower-ltl-to-core does not support temporal LTL operation 'ltl.until'}}
  %3 = ltl.until %1, %2 : !ltl.sequence, !ltl.sequence
  verif.assert %3 : !ltl.property
  hw.output
}

// -----

hw.module @UnclockedAtom(in %clock : !seq.clock, in %a : i1, in %b : i1) {
  %0 = seq.from_clock %clock
  %1 = ltl.clocked_atom %a, posedge %0 : i1
  // expected-error @below {{lower-ltl-to-core does not support block argument LTL properties}}
  %2 = ltl.implication %1, %b : !ltl.sequence, i1
  verif.assert %2 : !ltl.property
  hw.output
}

// -----

hw.module @UnclockedAntecedent(in %clock : !seq.clock, in %a : i1, in %b : i1) {
  %0 = seq.from_clock %clock
  %1 = ltl.clocked_atom %b, posedge %0 : i1
  // expected-error @below {{lower-ltl-to-core does not support block argument LTL sequences}}
  %2 = ltl.implication %a, %1 : i1, !ltl.sequence
  verif.assert %2 : !ltl.property
  hw.output
}

// -----

hw.module @LengthOverflow(in %clock : !seq.clock, in %a : i1) {
  %0 = seq.from_clock %clock
  %1 = ltl.clocked_atom %a, posedge %0 : i1
  // expected-error @below {{LTL sequence length overflows 64 bits}}
  %2 = ltl.clocked_delay %1, posedge %0, 18446744073709551615, 1 : !ltl.sequence
  verif.assert %2 : !ltl.sequence
  hw.output
}

// -----

// expected-error @below {{lower-ltl-to-core cannot eliminate LTL type '!ltl.property'}}
hw.module @LTLTypedPort(in %property : !ltl.property) {
  hw.output
}

// -----

// expected-error @below {{lower-ltl-to-core cannot eliminate LTL type '!hw.array<2x!ltl.sequence>'}}
hw.module @NestedLTLTypedPort(in %sequences : !hw.array<2x!ltl.sequence>) {
  hw.output
}

// -----

hw.module @LTLTypeAttribute(in %a : i1) {
  // expected-error @below {{lower-ltl-to-core cannot eliminate LTL type '!ltl.property' in attribute 'test.ltl_type'}}
  %0 = hw.wire %a {test.ltl_type = !ltl.property} : i1
  hw.output
}

// -----

hw.module @BooleanImplicationWithProperty(
    in %a : i1, in %property : !ltl.property) {
  // expected-error @below {{failed to legalize operation 'ltl.implication' that was explicitly marked illegal}}
  %0 = ltl.implication %a, %property : i1, !ltl.property
  verif.assert %0 : !ltl.property
  hw.output
}

// -----

hw.module @BooleanNotWithProperty(in %property : !ltl.property) {
  // expected-error @below {{failed to legalize operation 'ltl.not' that was explicitly marked illegal}}
  %0 = ltl.not %property : !ltl.property
  verif.assert %0 : !ltl.property
  hw.output
}

// -----

hw.module @BooleanAndWithProperty(
    in %a : i1, in %property : !ltl.property) {
  // expected-error @below {{failed to legalize operation 'ltl.and' that was explicitly marked illegal}}
  %0 = ltl.and %a, %property : i1, !ltl.property
  verif.assert %0 : !ltl.property
  hw.output
}

// -----

hw.module @BooleanOrWithProperty(
    in %a : i1, in %property : !ltl.property) {
  // expected-error @below {{failed to legalize operation 'ltl.or' that was explicitly marked illegal}}
  %0 = ltl.or %a, %property : i1, !ltl.property
  verif.assert %0 : !ltl.property
  hw.output
}

// -----

hw.module @BooleanIntersectWithSequence(
    in %a : i1, in %sequence : !ltl.sequence) {
  // expected-error @below {{failed to legalize operation 'ltl.intersect' that was explicitly marked illegal}}
  %0 = ltl.intersect %a, %sequence : i1, !ltl.sequence
  verif.assert %0 : !ltl.sequence
  hw.output
}

// -----

hw.module @BooleanNotOfPropertyResult(in %a : i1, in %b : i1) {
  %0 = ltl.implication %a, %b : i1, i1
  // expected-error @below {{failed to legalize operation 'ltl.not' that was explicitly marked illegal}}
  %1 = ltl.not %0 : !ltl.property
  hw.output
}

// -----

hw.module @BooleanAndOfPropertyResult(in %a : i1) {
  %0 = ltl.not %a : i1
  // expected-error @below {{failed to legalize operation 'ltl.and' that was explicitly marked illegal}}
  %1 = ltl.and %0, %0 : !ltl.property, !ltl.property
  hw.output
}

// -----

hw.module @UnsupportedEventually(in %a : i1) {
  // expected-error @below {{failed to legalize operation 'ltl.eventually' that was explicitly marked illegal}}
  %0 = ltl.eventually %a : i1
  verif.assert %0 : !ltl.property
  hw.output
}

// -----

hw.module @TemporalClockedAssert(in %clock : !seq.clock, in %a : i1) {
  %0 = seq.from_clock %clock
  // expected-error @below {{failed to legalize operation 'ltl.clocked_atom' that was explicitly marked illegal}}
  %1 = ltl.clocked_atom %a, posedge %0 : i1
  verif.clocked_assert %1, posedge %0 : !ltl.sequence
  hw.output
}
