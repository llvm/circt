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

//  Check that ignore-asserts-until works correctly
//  RUN: circt-bmc %s -b 7 --ignore-asserts-until=4 --module Counter --shared-libs=%libz3 | FileCheck %s --check-prefix=COUNTER4
//  COUNTER4: Bound reached with no violations!
//  RUN: circt-bmc %s -b 7 --ignore-asserts-until=3 --module Counter --shared-libs=%libz3 | FileCheck %s --check-prefix=COUNTER3
//  COUNTER3: Assertion can be violated!

hw.module @Counter(in %clk: !seq.clock, out count: i3) {
  %init = seq.initial () {
    %c0_i3 = hw.constant 0 : i3
    seq.yield %c0_i3 : i3
  } : () -> !seq.immutable<i3>
  %c1_i3 = hw.constant 1 : i3
  %regPlusOne = comb.add %reg, %c1_i3 : i3
  %reg = seq.compreg %regPlusOne, %clk initial %init : i3
  // Condition - count is always greater than 3
  %c3_i3 = hw.constant 3 : i3
  %lt = comb.icmp ugt %reg, %c3_i3 : i3
  verif.assert %lt : i1
  hw.output %reg : i3
}

// Check that reset can be triggered
// RUN: circt-bmc %s -b 4 --module Counter_with_reset --shared-libs=%libz3 | FileCheck %s --check-prefix=REGRESET
// REGRESET: Assertion can be violated!
hw.module @Counter_with_reset(in %clk: !seq.clock, in %rst: i1, out count: i3) {
  %init = seq.initial () {
    %c1_i3 = hw.constant 1 : i3
    seq.yield %c1_i3 : i3
  } : () -> !seq.immutable<i3>
  %c0_i3 = hw.constant 0 : i3
  %c1_i3 = hw.constant 1 : i3
  %regPlusOne = comb.add %reg, %c1_i3 : i3
  %reg = seq.compreg %regPlusOne, %clk reset %rst, %c0_i3 initial %init : i3
  %neq = comb.icmp bin ne %reg, %c0_i3 : i3
  // Assertion will be violated if the reset is triggered
  verif.assert %neq : i1
  hw.output %reg : i3
}

// Check that reset of firreg can be triggered
// RUN: circt-bmc %s -b 4 --module Counter_with_firreg_sync_reset --shared-libs=%libz3 | FileCheck %s --check-prefix=FIRREGRESET
// FIRREGRESET: Assertion can be violated!
hw.module @Counter_with_firreg_sync_reset(in %clk: !seq.clock, in %rst: i1, out count: i3) {
  %c0_i3 = hw.constant 0 : i3
  %c1_i3 = hw.constant 1 : i3
  %regPlusOne = comb.add %reg, %c1_i3 : i3
  %reg = seq.firreg %regPlusOne clock %clk reset sync %rst, %c0_i3 preset 1 : i3
  %neq = comb.icmp bin ne %reg, %c0_i3 : i3
  // Assertion will be violated if the reset is triggered
  verif.assert %neq : i1
  hw.output %reg : i3
}

