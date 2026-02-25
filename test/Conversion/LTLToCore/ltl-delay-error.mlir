// RUN: circt-opt %s --lower-ltl-to-core --verify-diagnostics

hw.module @ltl_bad(in %clk: !seq.clock, in %req: i1, in %ack: i1) attributes {num_regs = 0 : i32, initial_values = []} {
  // Unbounded delay is not supported by lower-ltl-to-core.
  %seq = ltl.delay %ack, 2 : i1
  %imp = ltl.implication %req, %seq : i1, !ltl.sequence
  %clk_i1 = seq.from_clock %clk
  %prop = ltl.clock %imp, posedge %clk_i1 : !ltl.property
  // expected-error @below {{unsupported LTL delay pattern for LowerLTLToCore}}
  // expected-error @below {{failed to legalize operation 'verif.assert' that was explicitly marked illegal}}
  verif.assert %prop : !ltl.property
  hw.output
}
