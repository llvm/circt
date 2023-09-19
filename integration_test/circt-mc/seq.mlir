// These tests will be only enabled if circt-mc is built.
// REQUIRES: circt-mc

//  RUN: circt-mc %s -b 10 --module ClkProp | FileCheck %s --check-prefix=CLKPROP
//  CLKPROP: Success!

hw.module @ClkProp(%clk: !seq.clock, %i0: i1) {
  %reg = seq.compreg %i0, %clk : i1
  // Condition (equivalent to %clk -> %reg == %i0)
  %c-1_i1 = hw.constant -1 : i1
  %clk_i1 = seq.from_clock %clk
  %nclk = comb.xor bin %clk_i1, %c-1_i1 : i1
  %eq = comb.icmp bin eq %i0, %reg : i1
  %imp = comb.or bin %nclk, %eq : i1
  verif.assert %imp : i1
}
