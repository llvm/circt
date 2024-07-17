// RUN: circt-opt --sim-proceduralize --split-input-file --verify-diagnostics %s

hw.module @cyclic_concat(in %clk : !seq.clock) {
  %true = hw.constant true
  %ping = sim.fmt.concat (%pong)
  %pong = sim.fmt.concat (%ping)
  // expected-error @below {{Cyclic format string cannot be proceduralized.}}
  sim.print %ping on %clk if %true
}
