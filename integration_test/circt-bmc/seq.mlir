// REQUIRES: libz3
// REQUIRES: circt-bmc-jit

// Check propagation of state through comb ops
//  RUN: circt-bmc %s -b 10 --module StateProp --shared-libs=%libz3 | FileCheck %s --check-prefix=STATEPROP
//  STATEPROP: Bound reached with no violations!

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

// RUN: circt-bmc %s -b 10 --module aggregateReg --shared-libs=%libz3 | FileCheck %s --check-prefix=AGGREGATEREG
// Can be violated because the register does not have an initial value specified.
// TODO: add support for aggregate initial values and adjust this test accordingly.
// AGGREGATEREG: Assertion can be violated!
hw.module @aggregateReg(in %clk: !seq.clock) {
  %c0 = hw.constant 0 : i32
  %arr = hw.array_create %c0, %c0 : i32
  %res = seq.compreg %arr, %clk : !hw.array<2xi32>
  %idx = hw.constant 0 : i1
  %get = hw.array_get %res[%idx] : !hw.array<2xi32>, i1
  %eq = comb.icmp bin eq %c0, %get : i32
  verif.assert %eq : i1
}
