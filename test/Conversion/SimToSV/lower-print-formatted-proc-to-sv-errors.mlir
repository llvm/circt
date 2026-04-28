// RUN: circt-opt --lower-sim-to-sv --split-input-file --verify-diagnostics %s

hw.module @unsupported_padding_char(in %clk : i1, in %arg : i8) {
  hw.triggered posedge %clk (%arg) : i8 {
    ^bb0(%arg_in : i8):
    %lit = sim.fmt.literal "bad="
    // expected-error @below {{sim.fmt.dec only supports paddingChar 32 (' ') or 48 ('0')}}
    %bad = sim.fmt.dec %arg_in paddingChar 42 specifierWidth 2 : i8
    // expected-note @below {{while lowering format string}}
    %msg = sim.fmt.concat (%lit, %bad)
    // expected-error @below {{cannot lower 'sim.proc.print' to SystemVerilog}}
    sim.proc.print %msg
  }
}

// -----

hw.module @unsupported_input_block_argument(in %clk : i1, in %arg : !sim.fstring) {
  hw.triggered posedge %clk (%arg) : !sim.fstring {
    // expected-error @below {{block argument format strings are unsupported}}
    // expected-note @below {{while lowering format string}}
    ^bb0(%arg_in : !sim.fstring):
    // expected-error @below {{cannot lower 'sim.proc.print' to SystemVerilog}}
    sim.proc.print %arg_in
  }
}

// -----

hw.module @unsupported_get_file_padding_char(in %clk : i1, in %arg : i8) {
  hw.triggered posedge %clk (%arg) : i8 {
    ^bb0(%arg_in : i8):
    %filePrefix = sim.fmt.literal "bad="
    // expected-error @below {{sim.fmt.dec only supports paddingChar 32 (' ') or 48 ('0')}}
    %fileValue = sim.fmt.dec %arg_in paddingChar 42 specifierWidth 2 : i8
    // expected-note @below {{while lowering file name}}
    %fileName = sim.fmt.concat (%filePrefix, %fileValue)
    // expected-error @below {{cannot lower 'sim.get_file' to SystemVerilog}}
    %file = sim.get_file %fileName
    %msg = sim.fmt.literal "message"
    sim.proc.print %msg to %file
  }
}
