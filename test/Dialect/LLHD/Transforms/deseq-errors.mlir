// RUN: circt-opt --llhd-deseq --verify-diagnostics %s

// A module input cannot be represented as a seq.firreg preset. Report the
// unsupported initializer rather than silently dropping it.
hw.module @NonconstantInit(in %clock: i1, in %d: i42) {
  %zero = hw.constant 0 : i42
  %time = llhd.constant_time <0ns, 1d, 0e>
  %value, %enable = llhd.process -> i42, i1 {
    %true = hw.constant true
    %false = hw.constant false
    cf.br ^bb1(%zero, %false : i42, i1)
  ^bb1(%data: i42, %en: i1):
    llhd.wait yield (%data, %en : i42, i1), (%clock : i1), ^bb2(%clock : i1)
  ^bb2(%prev: i1):
    %notPrev = comb.xor bin %prev, %true : i1
    %posedge = comb.and bin %notPrev, %clock : i1
    cf.cond_br %posedge, ^bb1(%d, %true : i42, i1), ^bb1(%zero, %false : i42, i1)
  }
  // expected-error@+1 {{cannot lower a nonconstant signal initializer to a register preset}}
  %sig = llhd.sig %d : <i42>
  llhd.drv %sig, %value after %time if %enable : i42
}
