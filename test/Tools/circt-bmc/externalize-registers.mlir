// RUN: circt-opt --externalize-registers %s | FileCheck %s

// CHECK:  hw.module @comb(in [[IN0:%.+]] : i32, in [[IN1:%.+]] : i32, out {{.+}} : i32) attributes {num_regs = 0 : i32} {
// CHECK:    [[ADD:%.+]] = comb.add [[IN0]], [[IN1]]
// CHECK:    hw.output [[ADD]]
// CHECK:  }
hw.module @comb(in %in0: i32, in %in1: i32, out out: i32) {
  %0 = comb.add %in0, %in1 : i32
  hw.output %0 : i32
}

// CHECK:  hw.module @one_reg(in [[CLK:%.+]] : !seq.clock, in [[IN0:%.+]] : i32, in [[IN1:%.+]] : i32, in [[OLD_REG:%.+]] : i32, out {{.+}} : i32, out {{.+}} : i32) attributes {num_regs = 1 : i32} {
// CHECK:    [[ADD:%.+]] = comb.add [[IN0]], [[IN1]]
// CHECK:    hw.output [[OLD_REG]], [[ADD]]
// CHECK:  }
hw.module @one_reg(in %clk: !seq.clock, in %in0: i32, in %in1: i32, out out: i32) {
  %0 = comb.add %in0, %in1 : i32
  %1 = seq.compreg %0, %clk : i32
  hw.output %1 : i32
}

// CHECK:  hw.module @two_reg(in [[CLK:%.+]] : !seq.clock, in [[IN0:%.+]] : i32, in [[IN1:%.+]] : i32, in [[OLD_REG0:%.+]] : i32, in [[OLD_REG1:%.+]] : i32, out {{.+}} : i32, out {{.+}} : i32, out {{.+}} : i32) attributes {num_regs = 2 : i32} {
// CHECK:    [[ADD:%.+]] = comb.add [[IN0]], [[IN1]]
// CHECK:    hw.output [[OLD_REG1]], [[ADD]], [[OLD_REG0]]
// CHECK:  }
hw.module @two_reg(in %clk: !seq.clock, in %in0: i32, in %in1: i32, out out: i32) {
  %0 = comb.add %in0, %in1 : i32
  %1 = seq.compreg %0, %clk : i32
  %2 = seq.compreg %1, %clk : i32
  hw.output %2 : i32
}

// CHECK:  hw.module @named_regs(in [[CLK:%.+]] : !seq.clock, in [[IN0:%.+]] : i32, in [[IN1:%.+]] : i32, in [[OLD_REG0:%.+]] : i32, in [[OLD_REG1:%.+]] : i32, out {{.+}} : i32, out {{.+}} : i32, out {{.+}} : i32) attributes {num_regs = 2 : i32} {
// CHECK:    [[ADD:%.+]] = comb.add [[IN0]], [[IN1]]
// CHECK:    hw.output [[OLD_REG1]], [[ADD]], [[OLD_REG0]]
// CHECK:  }
hw.module @named_regs(in %clk: !seq.clock, in %in0: i32, in %in1: i32, out out: i32) {
  %0 = comb.add %in0, %in1 : i32
  %1 = seq.compreg sym @reg1 %0, %clk : i32
  %2 = seq.compreg sym @reg2 %1, %clk : i32
  hw.output %2 : i32
}

// CHECK:  hw.module @nested_reg(in [[CLK:%.+]] : !seq.clock, in [[IN0:%.+]] : i32, in [[IN1:%.+]] : i32, in [[OLD_REG:%.+]] : i32, out {{.+}} : i32, out {{.+}} : i32) attributes {num_regs = 1 : i32} {
// CHECK:    [[INSTOUT:%.+]], [[INSTREG:%.+]] = hw.instance "one_reg" @one_reg(clk: [[CLK]]: !seq.clock, in0: [[IN0]]: i32, in1: [[IN1]]: i32, _0: [[OLD_REG]]: i32) -> ({{.+}}: i32, {{.+}}: i32)
// CHECK:    hw.output [[INSTOUT]], [[INSTREG]]
// CHECK:  }
hw.module @nested_reg(in %clk: !seq.clock, in %in0: i32, in %in1: i32, out out: i32) {
  %0 = hw.instance "one_reg" @one_reg(clk: %clk: !seq.clock, in0: %in0: i32, in1: %in1: i32) ->  (out: i32)
  hw.output %0 : i32
}

// CHECK:  hw.module @nested_nested_reg(in [[CLK:%.+]] : !seq.clock, in [[IN0:%.+]] : i32, in [[IN1:%.+]] : i32, in [[OLD_REG0:%.+]] : i32, in [[OLD_REG1:%.+]] : i32, out {{.+}} : i32, out {{.+}} : i32, out {{.+}} : i32) attributes {num_regs = 2 : i32} {
// CHECK:    [[INSTOUT:%.+]], [[INSTREG:%.+]] = hw.instance "nested_reg" @nested_reg(clk: [[CLK]]: !seq.clock, in0: [[IN0]]: i32, in1: [[IN1]]: i32, _0: [[OLD_REG0]]: i32) -> ({{.+}}: i32, {{.+}}: i32)
// CHECK:    hw.output [[OLD_REG1]], [[INSTREG]], [[INSTOUT]]
// CHECK:  }
hw.module @nested_nested_reg(in %clk: !seq.clock, in %in0: i32, in %in1: i32, out out: i32) {
  %0 = hw.instance "nested_reg" @nested_reg(clk: %clk: !seq.clock, in0: %in0: i32, in1: %in1: i32) ->  (out: i32)
  %1 = seq.compreg %0, %clk : i32
  hw.output %1 : i32
}
