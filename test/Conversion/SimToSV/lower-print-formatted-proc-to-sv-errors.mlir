// RUN: circt-opt --lower-sim-to-sv --split-input-file --verify-diagnostics %s

hw.module @unsupported_padding_char(in %clk : i1, in %arg : i8) {
  hw.triggered posedge %clk (%arg) : i8 {
    ^bb0(%arg_in : i8):
    %lit = sim.fmt.literal "bad="
    // expected-error @below {{cannot lower format string for 'sim.proc.print' to SystemVerilog: sim.fmt.dec only supports paddingChar 32 (' ') or 48 ('0')}}
    %bad = sim.fmt.dec %arg_in paddingChar 42 specifierWidth 2 : i8
    %msg = sim.fmt.concat (%lit, %bad)
    sim.proc.print %msg
  }
}

// -----

hw.module @unsupported_input_block_argument(in %clk : i1, in %arg : !sim.fstring) {
  hw.triggered posedge %clk (%arg) : !sim.fstring {
    // expected-error @below {{cannot lower format string for 'sim.proc.print' to SystemVerilog: block argument format strings are unsupported}}
    ^bb0(%arg_in : !sim.fstring):
    sim.proc.print %arg_in
  }
}

// -----

hw.module @unsupported_get_file_padding_char(in %clk : i1, in %arg : i8) {
  hw.triggered posedge %clk (%arg) : i8 {
    ^bb0(%arg_in : i8):
    %filePrefix = sim.fmt.literal "bad="
    // expected-error @below {{cannot lower file name for 'sim.get_file' to SystemVerilog: sim.fmt.dec only supports paddingChar 32 (' ') or 48 ('0')}}
    %fileValue = sim.fmt.dec %arg_in paddingChar 42 specifierWidth 2 : i8
    %fileName = sim.fmt.concat (%filePrefix, %fileValue)
    %file = sim.get_file %fileName
    %msg = sim.fmt.literal "message"
    sim.proc.print %msg to %file
  }
}
