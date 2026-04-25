// RUN: circt-opt --lower-sim-to-sv --split-input-file --verify-diagnostics %s

hw.module @unsupported_padding_char(in %clk : i1, in %arg : i8) {
  hw.triggered posedge %clk (%arg) : i8 {
    ^bb0(%arg_in : i8):
    %lit = sim.fmt.literal "bad="
    // expected-error @below {{cannot lower 'sim.proc.print' to sv.write: sim.fmt.dec only supports paddingChar 32 (' ') or 48 ('0') for SystemVerilog lowering}}
    %bad = sim.fmt.dec %arg_in paddingChar 42 specifierWidth 2 : i8
    %msg = sim.fmt.concat (%lit, %bad)
    sim.proc.print %msg
  }
}

// -----

hw.module @unsupported_input_block_argument(in %clk : i1, in %arg : !sim.fstring) {
  hw.triggered posedge %clk (%arg) : !sim.fstring {
    // expected-error @below {{cannot lower 'sim.proc.print' to sv.write: block argument format strings are unsupported as sim.proc.print input}}
    ^bb0(%arg_in : !sim.fstring):
    sim.proc.print %arg_in
  }
}

// -----

hw.module @unsupported_stream_block_argument(
    in %clk : i1, in %stream : !sim.output_stream) {
  hw.triggered posedge %clk (%stream) : !sim.output_stream {
    ^bb0(%stream_in : !sim.output_stream):
    %fmt = sim.fmt.literal "x"
    // expected-error @below {{failed to legalize unresolved materialization from ('!sim.output_stream') to ('i32') that remained live after conversion}}
    // expected-note @below {{see existing live user here: sv.fwrite %0, "x"}}
    sim.proc.print %fmt to %stream_in
  }
}

// -----

hw.module @unsupported_file_stream(
    in %clk : i1) {
  hw.triggered posedge %clk {
    %fmt = sim.fmt.literal "x"
    // expected-error @below {{failed to legalize operation 'sim.fmt.literal' that was explicitly marked illegal: %0 = "sim.fmt.literal"() <{literal = "file.txt"}> : () -> !sim.fstring}}
    %file = sim.fmt.literal "file.txt"
    %stream_in = sim.get_file %file
    sim.proc.print %fmt to %stream_in
  }
}
