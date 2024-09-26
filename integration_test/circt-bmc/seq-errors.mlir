// REQUIRES: libz3
// REQUIRES: circt-bmc-jit

//  RUN: circt-bmc %s -b 10 --module Counter --shared-libs=%libz3 | FileCheck %s --check-prefix=COUNTER
//  COUNTER: Assertion can be violated!

hw.module @Counter(in %clk: !seq.clock, out count: i2) {
  %c1_i2 = hw.constant 1 : i2
  %regPlusOne = comb.add %reg, %c1_i2 : i2
  %reg = seq.compreg %regPlusOne, %clk : i2
  // Condition - count should never reach 3 (deliberately not true)
  // FIXME: add an initial condition here once we support them, currently it
  // can be violated on the first cycle as 3 is a potential initial value.
  // Can also use this to check bounds are behaving as expected.
  %c3_i2 = hw.constant 3 : i2
  %lt = comb.icmp ult %reg, %c3_i2 : i2
  verif.assert %lt : i1
  hw.output %reg : i2
}
