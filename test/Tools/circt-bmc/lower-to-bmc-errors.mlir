// RUN: circt-opt --lower-to-bmc="top-module=testModule bound=10" --split-input-file --verify-diagnostics %s

// expected-error @below {{hw.module named 'testModule' not found}}
module {}

// -----

// expected-error @below {{hw.module named 'testModule' not found}}
module {
  func.func @testModule(%in: i8) -> (i8) {
    func.return %in : i8
  }
}

// -----

// expected-error @below {{no property provided to check in module}}
hw.module @testModule(in %in0 : i32, in %in1 : i32, out out : i32) attributes {num_regs = 0 : i32, initial_values = []} {
  %0 = comb.add %in0, %in1 : i32
  hw.output %0 : i32
}

// -----

// expected-error @below {{no num_regs or initial_values attribute found - please run externalize registers pass first}}
hw.module @testModule(in %in0 : i32, in %in1 : i32, out out : i32) {
  %0 = comb.add %in0, %in1 : i32
  %1 = comb.icmp eq %in0, %in1 : i32
  verif.assert %1 : i1
  hw.output %0 : i32
}

// -----

// expected-error @below {{no num_regs or initial_values attribute found - please run externalize registers pass first}}
hw.module @testModule(in %in0 : i32, in %in1 : i32, out out : i32) attributes {num_regs = 0 : i32} {
  %0 = comb.add %in0, %in1 : i32
  %1 = comb.icmp eq %in0, %in1 : i32
  verif.assert %1 : i1
  hw.output %0 : i32
}

// -----

// expected-error @below {{no num_regs or initial_values attribute found - please run externalize registers pass first}}
hw.module @testModule(in %in0 : i32, in %in1 : i32, out out : i32) attributes {initial_values = []} {
  %0 = comb.add %in0, %in1 : i32
  %1 = comb.icmp eq %in0, %in1 : i32
  verif.assert %1 : i1
  hw.output %0 : i32
}

// -----

// expected-error @below {{initial_values attribute must contain only integer or unit attributes}}
hw.module @testModule(in %clk0 : !seq.clock, in %in0 : i32, in %in1 : i32, in %reg0_state : i32, in %reg1_state : i32, out out : i32, out reg0_input : i32, out reg1_input : i32) attributes {num_regs = 2 : i32, initial_values = [unit, "foo"]} {
  %0 = comb.add %reg0_state, %reg1_state : i32
  %1 = comb.icmp eq %0, %in0 : i32
  verif.assert %1 : i1
  hw.output %0, %in0, %in1 : i32, i32, i32
}

// -----

// expected-error @below {{designs with multiple clocks not yet supported}}
hw.module @testModule(in %clk0 : !seq.clock, in %clk1 : !seq.clock, in %in0 : i32, in %in1 : i32, in %reg0_state : i32, in %reg1_state : i32, out out : i32, out reg0_input : i32, out reg1_input : i32) attributes {num_regs = 2 : i32, initial_values = [unit, unit]} {
  %0 = comb.add %reg0_state, %reg1_state : i32
  %1 = comb.icmp eq %0, %in0 : i32
  verif.assert %1 : i1
  hw.output %0, %in0, %in1 : i32, i32, i32
}

// -----

// expected-error @below {{designs with multiple clocks not yet supported}}
hw.module @testModule(in %clk0 : !seq.clock, in %clkStruct : !hw.struct<clk: !seq.clock>, in %in0 : i32, in %in1 : i32, in %reg0_state : i32, in %reg1_state : i32, out out : i32, out reg0_input : i32, out reg1_input : i32) attributes {num_regs = 2 : i32, initial_values = [unit, unit]} {
  %0 = comb.add %reg0_state, %reg1_state : i32
  %1 = comb.icmp eq %0, %in0 : i32
  verif.assert %1 : i1
  hw.output %0, %in0, %in1 : i32, i32, i32
}

// -----

// expected-error @below {{could not resolve cycles in module}}
hw.module @testModule(in %in0 : i32) attributes {num_regs = 0 : i32, initial_values = []} {
  %add = comb.add %in0, %or : i32
  %or = comb.or %in0, %add : i32
  // Dummy property
  %true = hw.constant true
  verif.assert %true : i1
  hw.output
}
