// RUN: circt-opt --pass-pipeline='builtin.module(hw.module(arc-lower-hw-triggered))' --verify-diagnostics --split-input-file %s

hw.module @unsupported_body(in %clk : !seq.clock, in %value : !sim.fstring) {
  %trg = seq.from_clock %clk
  hw.triggered posedge %trg(%value) : !sim.fstring {
  ^bb0(%arg0 : !sim.fstring):
    // expected-error @below {{Sim dialect ops inside hw.triggered are unsupported by this Arc lowering proof-of-concept}}
    sim.proc.print %arg0
  }
  hw.output
}

// -----

hw.module @unsupported_nested_body(in %clk : !seq.clock, in %cond : i1,
                                   in %value : !sim.fstring) {
  %trg = seq.from_clock %clk
  hw.triggered posedge %trg(%cond, %value) : i1, !sim.fstring {
  ^bb0(%arg0 : i1, %arg1 : !sim.fstring):
    scf.if %arg0 {
      // expected-error @below {{Sim dialect ops inside hw.triggered are unsupported by this Arc lowering proof-of-concept}}
      sim.proc.print %arg1
    }
  }
  hw.output
}
