// RUN: circt-opt --externalize-registers --lower-to-bmc="top-module=two_clk_design bound=10" %s | FileCheck %s

// Verify that multi-clock designs are handled end-to-end: each clock gets an
// independent toggle in the init/loop regions of verif.bmc.

// CHECK: func.func @two_clk_design() {
// CHECK:   [[BMC:%.+]] = verif.bmc bound 20 num_regs 2 initial_values [unit, unit] init {
// CHECK:     [[FALSE0:%.+]] = hw.constant false
// CHECK:     [[CLK0_INIT:%.+]] = seq.to_clock [[FALSE0]]
// CHECK:     [[FALSE1:%.+]] = hw.constant false
// CHECK:     [[CLK1_INIT:%.+]] = seq.to_clock [[FALSE1]]
// CHECK:     verif.yield [[CLK0_INIT]], [[CLK1_INIT]]
// CHECK:   } loop {
// CHECK:   ^bb0([[CLK0:%.+]]: !seq.clock, [[CLK1:%.+]]: !seq.clock):
// CHECK:     [[FROM0:%.+]] = seq.from_clock [[CLK0]]
// CHECK:     [[TRUE0:%.+]] = hw.constant true
// CHECK:     [[NCLK0:%.+]] = comb.xor [[FROM0]], [[TRUE0]]
// CHECK:     [[NEW_CLK0:%.+]] = seq.to_clock [[NCLK0]]
// CHECK:     [[FROM1:%.+]] = seq.from_clock [[CLK1]]
// CHECK:     [[TRUE1:%.+]] = hw.constant true
// CHECK:     [[NCLK1:%.+]] = comb.xor [[FROM1]], [[TRUE1]]
// CHECK:     [[NEW_CLK1:%.+]] = seq.to_clock [[NCLK1]]
// CHECK:     verif.yield [[NEW_CLK0]], [[NEW_CLK1]]
// CHECK:   } circuit {
// CHECK:   ^bb0([[C0:%.+]]: !seq.clock, [[C1:%.+]]: !seq.clock,

hw.module @two_clk_design(in %clk0: !seq.clock, in %clk1: !seq.clock,
                           in %in: i32, out out: i32) {
  %reg0 = seq.compreg %in,   %clk0 : i32
  %reg1 = seq.compreg %reg0, %clk1 : i32
  %prop = comb.icmp uge %reg1, %in : i32
  verif.assert %prop : i1
  hw.output %reg1 : i32
}
