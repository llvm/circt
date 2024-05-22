// These tests will be only enabled if circt-mc is built.

//  RUN: circt-bmc %s -b 10 --module ClkProp | FileCheck %s --check-prefix=CLKPROP
//  CLKPROP: Success!

hw.module @ClkProp(in %clk: !seq.clock, in %i0: i1) {
  %reg = seq.compreg %i0, %clk : i1
  // Condition (equivalent to %clk -> %reg == %i0)
  %c-1_i1 = hw.constant -1 : i1
  %clk_i1 = seq.from_clock %clk
  %nclk = comb.xor bin %clk_i1, %c-1_i1 : i1
  %eq = comb.icmp bin eq %i0, %reg : i1
  %imp = comb.or bin %nclk, %eq : i1
  verif.assert %imp : i1
}

// Check propagation of state through comb ops

//  RUN: circt-bmc %s -b 10 --module StateProp --shared-libs=%libz3 | FileCheck %s --check-prefix=STATEPROP
//  STATEPROP: Success!

hw.module @StateProp(in %clk: !seq.clock, in %i0: i1) {
  %c-1_i1 = hw.constant -1 : i1
  %reg = seq.compreg %i0, %clk : i1
  %not_reg = comb.xor bin %reg, %c-1_i1 : i1
  %not_not_reg = comb.xor bin %not_reg, %c-1_i1 : i1
  // Condition (equivalent to %clk -> %reg == %not_not_reg)
  %clk_i1 = seq.from_clock %clk
  %nclk = comb.xor bin %clk_i1, %c-1_i1 : i1
  %eq = comb.icmp bin eq %not_not_reg, %reg : i1
  %imp = comb.or bin %nclk, %eq : i1
  verif.assert %imp : i1
}
