// RUN: circt-opt %s --arc-lower-state  --split-input-file --verify-diagnostics

arc.define @DummyArc(%arg0: i42) -> i42 {
  arc.output %arg0 : i42
}

// expected-error @+1 {{Value cannot be used in initializer.}}
hw.module @argInit(in %clk: !seq.clock, in %input: i42) {
  %0 = arc.state @DummyArc(%0) clock %clk initial (%input : i42) latency 1 : (i42) -> i42
}


// -----


arc.define @DummyArc(%arg0: i42) -> i42 {
  arc.output %arg0 : i42
}

hw.module @argInit(in %clk: !seq.clock, in %input: i42) {
  // expected-error @+1 {{Value cannot be used in initializer.}}
  %0 = arc.state @DummyArc(%0) clock %clk latency 1 : (i42) -> i42
  %1 = arc.state @DummyArc(%1) clock %clk initial (%0 : i42) latency 1 : (i42) -> i42
}
