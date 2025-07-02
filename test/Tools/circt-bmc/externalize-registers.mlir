// RUN: circt-opt --externalize-registers %s | FileCheck %s

// CHECK:  hw.module @comb(in [[IN0:%.+]] : i32, in [[IN1:%.+]] : i32, out {{.+}} : i32) attributes {initial_values = [], num_regs = 0 : i32} {
// CHECK:    [[ADD:%.+]] = comb.add [[IN0]], [[IN1]]
// CHECK:    hw.output [[ADD]]
// CHECK:  }
hw.module @comb(in %in0: i32, in %in1: i32, out out: i32) {
  %0 = comb.add %in0, %in1 : i32
  hw.output %0 : i32
}

// CHECK:  hw.module @one_reg(in [[CLK:%.+]] : !seq.clock, in [[IN0:%.+]] : i32, in [[IN1:%.+]] : i32, in [[OLD_REG:%.+]] : i32, out {{.+}} : i32, out {{.+}} : i32) attributes {initial_values = [0 : i32], num_regs = 1 : i32} {
// CHECK:    [[ADD:%.+]] = comb.add [[IN0]], [[IN1]]
// CHECK:    [[INITIAL:%.+]] = seq.initial() {
// CHECK:      [[C0_I32:%.+]] = hw.constant 0 : i32
// CHECK:      seq.yield [[C0_I32]]
// CHECK:    }
// CHECK:    hw.output [[OLD_REG]], [[ADD]]
// CHECK:  }
hw.module @one_reg(in %clk: !seq.clock, in %in0: i32, in %in1: i32, out out: i32) {
  %0 = comb.add %in0, %in1 : i32
  %1 = seq.initial() {
    %c0_i32 = hw.constant 0 : i32
    seq.yield %c0_i32 : i32
  } : () -> !seq.immutable<i32>
  %single_reg = seq.compreg %0, %clk initial %1 : i32
  hw.output %single_reg : i32
}

// CHECK:  hw.module @two_reg(in [[CLK:%.+]] : !seq.clock, in [[IN0:%.+]] : i32, in [[IN1:%.+]] : i32, in [[OLD_REG0:%.+]] : i32, in [[OLD_REG1:%.+]] : i32, out {{.+}} : i32, out {{.+}} : i32, out {{.+}} : i32) attributes {initial_values = [unit, unit], num_regs = 2 : i32} {
// CHECK:    [[ADD:%.+]] = comb.add [[IN0]], [[IN1]]
// CHECK:    hw.output [[OLD_REG1]], [[ADD]], [[OLD_REG0]]
// CHECK:  }
hw.module @two_reg(in %clk: !seq.clock, in %in0: i32, in %in1: i32, out out: i32) {
  %0 = comb.add %in0, %in1 : i32
  %1 = seq.compreg %0, %clk : i32
  %2 = seq.compreg %1, %clk : i32
  hw.output %2 : i32
}

// CHECK:  hw.module @named_regs(in [[CLK:%.+]] : !seq.clock, in [[IN0:%.+]] : i32, in [[IN1:%.+]] : i32, in %firstreg_state : i32, in %secondreg_state : i32, out {{.+}} : i32, out {{.+}} : i32, out {{.+}} : i32) attributes {initial_values = [unit, unit], num_regs = 2 : i32} {
// CHECK:    [[ADD:%.+]] = comb.add [[IN0]], [[IN1]]
// CHECK:    hw.output %secondreg_state, [[ADD]], %firstreg_state
// CHECK:  }
hw.module @named_regs(in %clk: !seq.clock, in %in0: i32, in %in1: i32, out out: i32) {
  %0 = comb.add %in0, %in1 : i32
  %firstreg = seq.compreg %0, %clk : i32
  %secondreg = seq.compreg %firstreg, %clk : i32
  hw.output %secondreg : i32
}

// CHECK:  hw.module @nested_reg(in [[CLK:%.+]] : !seq.clock, in [[IN0:%.+]] : i32, in [[IN1:%.+]] : i32, in [[OLD_REG:%.+]] : i32, out {{.+}} : i32, out {{.+}} : i32) attributes {initial_values = [0 : i32], num_regs = 1 : i32} {
// CHECK:    [[INSTOUT:%.+]], [[INSTREG:%.+]] = hw.instance "one_reg" @one_reg(clk: [[CLK]]: !seq.clock, in0: [[IN0]]: i32, in1: [[IN1]]: i32, {{.+}}: [[OLD_REG]]: i32) -> ({{.+}}: i32, {{.+}}: i32)
// CHECK:    hw.output [[INSTOUT]], [[INSTREG]]
// CHECK:  }
hw.module @nested_reg(in %clk: !seq.clock, in %in0: i32, in %in1: i32, out out: i32) {
  %0 = hw.instance "one_reg" @one_reg(clk: %clk: !seq.clock, in0: %in0: i32, in1: %in1: i32) ->  (out: i32)
  hw.output %0 : i32
}

// CHECK:  hw.module @nested_nested_reg(in [[CLK:%.+]] : !seq.clock, in [[IN0:%.+]] : i32, in [[IN1:%.+]] : i32, in %single_reg_state : i32, in %top_reg_state : i32, out {{.+}} : i32, out single_reg_input : i32, out top_reg_input : i32) attributes {initial_values = [0 : i32, unit], num_regs = 2 : i32} {
// CHECK:    [[INSTOUT:%.+]], [[INSTREG:%.+]] = hw.instance "nested_reg" @nested_reg(clk: [[CLK]]: !seq.clock, in0: [[IN0]]: i32, in1: [[IN1]]: i32, single_reg_state: %single_reg_state: i32) -> ({{.+}}: i32, single_reg_input: i32)
// CHECK:    hw.output %top_reg_state, [[INSTREG]], [[INSTOUT]]
// CHECK:  }
hw.module @nested_nested_reg(in %clk: !seq.clock, in %in0: i32, in %in1: i32, out out: i32) {
  %0 = hw.instance "nested_reg" @nested_reg(clk: %clk: !seq.clock, in0: %in0: i32, in1: %in1: i32) ->  (out: i32)
  %top_reg = seq.compreg %0, %clk : i32
  hw.output %top_reg : i32
}

// CHECK:  hw.module @different_initial_values(in [[CLK:%.+]] : !seq.clock, in [[IN:%.+]] : i32, in %reg0_state : i32, in %reg1_state : i32, in %reg2_state : i32, out reg0_input : i32, out reg1_input : i32, out reg2_input : i32) attributes {initial_values = [0 : i32, 42 : i32, unit], num_regs = 3 : i32} {
// CHECK:    [[INITIAL:%.+]]:2 = seq.initial() {
// CHECK:      [[C0_I32:%.+]] = hw.constant 0 : i32
// CHECK:      [[C42_I32:%.+]] = hw.constant 42 : i32
// CHECK:      seq.yield [[C0_I32]], [[C42_I32]]
// CHECK:    }
// CHECK:    hw.output [[IN]], [[IN]], [[IN]]
// CHECK:  }
hw.module @different_initial_values(in %clk: !seq.clock, in %in : i32) {
  %0:2 = seq.initial () {
    %c0_i32 = hw.constant 0 : i32
    %c42_i32 = hw.constant 42 : i32
    seq.yield %c0_i32, %c42_i32 : i32, i32
  } : () -> (!seq.immutable<i32>, !seq.immutable<i32>)
  %reg0 = seq.compreg %in, %clk initial %0#0 : i32
  %reg1 = seq.compreg %in, %clk initial %0#1  : i32
  %reg2 = seq.compreg %in, %clk : i32
  hw.output
}

// CHECK:  hw.module @reg_with_reset(in [[CLK:%.+]] : !seq.clock, in [[RST:%.+]] : i1, in [[IN:%.+]] : i32, in [[OLD_REG:%.+]] : i32, out {{.+}} : i32, out {{.+}} : i32) attributes {initial_values = [unit], num_regs = 1 : i32} {
// CHECK:    [[C0_I32:%.+]] = hw.constant 0 : i32
// CHECK:    [[MUX:%.+]] = comb.mux [[RST]], [[C0_I32]], [[IN]] : i32
// CHECK:    hw.output [[OLD_REG]], [[MUX]]
// CHECK:  }
hw.module @reg_with_reset(in %clk: !seq.clock, in %rst: i1, in %in: i32, out out: i32) {
  %c0_i32 = hw.constant 0 : i32
  %1 = seq.compreg %in, %clk reset %rst, %c0_i32 : i32
  hw.output %1 : i32
}
