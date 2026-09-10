// REQUIRES: libz3
// REQUIRES: circt-bmc-jit

//  RUN: circt-bmc %s -b 10 --module OrEqAnd --shared-libs=%libz3 | FileCheck %s --check-prefix=OREQAND
//  OREQAND: Assertion can be violated!

module {
  hw.module @OrEqAnd(in %i0: i1, in %i1: i1) {
    %or = comb.or bin %i0, %i1 : i1
    %and = comb.and bin %i0, %i1 : i1
    // Condition
    %cond = comb.icmp bin eq %or, %and : i1
    verif.assert %cond : i1
  }
}
