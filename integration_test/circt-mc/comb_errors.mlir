//  RUN: circt-mc %s -b 10 --module OrEqAnd --verify-diagnostics

// expected-error @+1 {{Properties do not hold on module.}}
module {
  hw.module @OrEqAnd(%i0: i1, %i1: i1) {
    %or = comb.or bin %i0, %i1 : i1
    %and = comb.and bin %i0, %i1 : i1
    // Condition
    %cond = comb.icmp bin eq %or, %and : i1
    verif.assert %cond : i1
  }
}