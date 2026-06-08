// RUN: circt-opt --lower-sim-to-sv --split-input-file --verify-diagnostics %s

hw.module @block_arg_scan_format(in %clk : i1, in %fd : i32) {
  hw.triggered posedge %clk (%fd) : i32 {
    // expected-error @below {{block argument scan strings are unsupported}}
    ^bb0(%fd_in : i32, %fmt : !sim.scan_string):
    %stream = sim.sv.channel_to_input_stream %fd_in
    // expected-error @below {{cannot lower 'sim.proc.scan' to SystemVerilog}}
    %count = sim.proc.scan %stream %fmt
  }
}
