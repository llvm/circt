//  RUN: circt-bmc %s -b 10 --module OrEqAnd --verify-diagnostics --shared-libs=%libz3

// expected-error @+1 {{Properties do not hold on module.}}
module {
  hw.module @OrEqAnd(in %i0: i1, in %i1: i1) {
    %or = comb.or bin %i0, %i1 : i1
    %and = comb.and bin %i0, %i1 : i1
    // Condition
    %cond = comb.icmp bin eq %or, %and : i1
    verif.assert %cond : i1
  }
}
