// RUN: circt-opt --externalize-registers --lower-to-bmc="top-module=twoClocks bound=10" --split-input-file --verify-diagnostics %s

// expected-error @below {{Designs with multiple clocks not yet supported.}}
hw.module @twoClocks(in %clk0: !seq.clock, in %clk1: !seq.clock, in %in0: i32, in %in1: i32, out out: i32) {
  %reg0 = seq.compreg %in0, %clk0 : i32
  %reg1 = seq.compreg %in1, %clk1 : i32
  %0 = comb.add %reg0, %reg1 : i32
  %prop = comb.icmp eq %0, %in0 : i32
  verif.assert %prop : i1
  hw.output %0 : i32
}
