// RUN: circt-opt --sim-proceduralize --split-input-file --verify-diagnostics %s

hw.module @cyclic_concat(in %clk : !seq.clock) {
  %true = hw.constant true
  %ping = sim.fmt.concat (%pong)
  %pong = sim.fmt.concat (%ping)
  // expected-error @below {{Cyclic format string cannot be proceduralized.}}
  sim.print %ping on %clk if %true
}

// -----

hw.module @stream_block_arg_unsupported(
  in %clk : !seq.clock,
  in %en : i1,
  in %stream : !sim.output_stream) {
  %msg = sim.fmt.literal "x"
  // expected-error @below {{proceduralization requires stream to be produced by sim.get_file, block arguments are unsupported}}
  sim.print %msg on %clk if %en to %stream
}
