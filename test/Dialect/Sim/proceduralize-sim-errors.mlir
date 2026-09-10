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

// -----

hw.module @leftover_value(
  in %clk : !seq.clock,
  in %en : i1,
  in %v : i8,
  out lit : !sim.fstring) {
  // expected-warning @below {{Operation has remaining (transitive) users outside of procedural regions.}}
  %foo = sim.fmt.literal "foo"
  // expected-warning @below {{Operation has remaining (transitive) users outside of procedural regions.}}
  %fmt = sim.fmt.hex %v, isUpper false specifierWidth 8 : i8
  // expected-warning @below {{Operation has remaining (transitive) users outside of procedural regions.}}
  %cat = sim.fmt.concat (%foo, %fmt)
  sim.print %cat on %clk if %en
  hw.output %cat : !sim.fstring
}
