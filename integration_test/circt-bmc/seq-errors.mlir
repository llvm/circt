// REQUIRES: libz3
// REQUIRES: circt-bmc-jit

//  Test with two bounds - one that doesn't run long enough for the counter to reach 3, and one that does
//  RUN: circt-bmc %s -b 2 --module Counter --shared-libs=%libz3 | FileCheck %s --check-prefix=COUNTER2
//  COUNTER2: Bound reached with no violations!
//  RUN: circt-bmc %s -b 10 --module Counter --shared-libs=%libz3 | FileCheck %s --check-prefix=COUNTER10
//  COUNTER10: Assertion can be violated!

hw.module @Counter(in %clk: !seq.clock, out count: i2) {
  %init = seq.initial () {
    %c0_i2 = hw.constant 0 : i2
    seq.yield %c0_i2 : i2
  } : () -> !seq.immutable<i2>
  %c1_i2 = hw.constant 1 : i2
  %regPlusOne = comb.add %reg, %c1_i2 : i2
  %reg = seq.compreg %regPlusOne, %clk initial %init : i2
  // Condition - count should never reach 3
  %c3_i2 = hw.constant 3 : i2
  %lt = comb.icmp ult %reg, %c3_i2 : i2
  verif.assert %lt : i1
  hw.output %reg : i2
}
