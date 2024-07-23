// RUN: circt-opt --llhd-temporal-code-motion --split-input-file --verify-diagnostics %s

hw.module @more_than_one_wait() {
  llhd.process {
    cf.br ^bb1
  ^bb1:
    // expected-note @below {{first 'llhd.wait' operation here}}
    llhd.wait ^bb2
  ^bb2:
    // expected-error @below {{only one 'llhd.wait' operation per process supported}}
    llhd.wait ^bb1
  }
}

// -----

hw.module @more_than_two_TRs() {
  // expected-error @below {{more than 2 temporal regions are currently not supported}}
  llhd.process {
    cf.br ^bb1
  ^bb1:
    llhd.wait ^bb2
  ^bb2:
    llhd.wait ^bb3
  ^bb3:
    llhd.wait ^bb1
  }
}

// -----

hw.module @more_than_one_TR_wait_terminator(in %cond: i1) {
  llhd.process {
    cf.br ^bb1
  ^bb1:
    cf.cond_br %cond, ^bb2, ^bb3
  ^bb2:
    // expected-error @below {{block with wait terminator has to be the only exiting block of that temporal region}}
    llhd.wait ^bb4
  ^bb3:
    // expected-note @below {{other block terminator in same TR here}}
    llhd.wait ^bb4
  ^bb4:
    cf.br ^bb1
  }
}
