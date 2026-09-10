// REQUIRES: libz3
// REQUIRES: circt-bmc-jit

//  Check that in rising clock mode, we never see a low clock.
//  RUN: circt-bmc %s -b 4 --module InputProp --shared-libs=%libz3 | FileCheck %s --check-prefix=ALLEDGES
//  ALLEDGES: Assertion can be violated!
//  RUN: circt-bmc %s -b 4 --module InputProp --shared-libs=%libz3 --rising-clocks-only | FileCheck %s --check-prefix=RISINGEDGES
//  RISINGEDGES: Bound reached with no violations!

hw.module @InputProp(in %clk: !seq.clock) {
  %fromclk = seq.from_clock %clk
  verif.assert %fromclk : i1
}
