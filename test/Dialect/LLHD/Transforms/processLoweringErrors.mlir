// RUN: circt-opt %s -llhd-process-lowering -split-input-file -verify-diagnostics

// Check wait with observing probed signals
hw.module @prbAndWaitNotObserved(inout %arg0 : i64) {
  llhd.process {
    cf.br ^bb1
  ^bb1:
    %0 = llhd.prb %arg0 : !hw.inout<i64>
    // expected-error @+1 {{during process-lowering: the wait terminator is required to have values used in the process as arguments}}
    llhd.wait ^bb1
  }
}

// -----

// Check that block arguments for the second block are not allowed.
hw.module @blockArgumentsNotAllowed(inout %arg0 : i64) {
  // expected-error @+1 {{during process-lowering: the second block (containing the llhd.wait) is not allowed to have arguments}}
  llhd.process {
    %prb = llhd.prb %arg0 : !hw.inout<i64>
    cf.br ^bb1(%prb : i64)
  ^bb1(%a : i64):
    llhd.wait ^bb1(%a: i64)
  }
}

// -----

// Check that the entry block is terminated by a cf.br terminator.
hw.module @entryBlockMustHaveBrTerminator() {
  // expected-error @+1 {{during process-lowering: the first block has to be terminated by a cf.br operation}}
  llhd.process {
    llhd.wait ^bb1
  ^bb1:
    llhd.wait ^bb1
  }
}

// -----

// Check that there is no optional time operand in the wait terminator.
hw.module @noOptionalTime() {
  llhd.process {
    cf.br ^bb1
  ^bb1:
    %time = llhd.constant_time #llhd.time<0ns, 0d, 0e>
    // expected-error @+1 {{during process-lowering: llhd.wait terminators with optional time argument cannot be lowered to structural LLHD}}
    llhd.wait for %time, ^bb1
  }
}

// -----

// Check that if there are two blocks, the second one is terminated by a wait terminator.
hw.module @secondBlockTerminatedByWait() {
  // expected-error @+1 {{during process-lowering: the second block must be terminated by llhd.wait}}
  llhd.process {
    cf.br ^bb1
  ^bb1:
    llhd.halt
  }
}

// -----

// Check that there are not more than two blocks.
hw.module @moreThanTwoBlocksNotAllowed() {
  // expected-error @+1 {{process-lowering only supports processes with either one basic block terminated by a llhd.halt operation or two basic blocks where the first one contains a cf.br terminator and the second one is terminated by a llhd.wait operation}}
  llhd.process {
    cf.br ^bb1
  ^bb1:
    cf.br ^bb2
  ^bb2:
    llhd.wait ^bb1
  }
}

// -----

hw.module @muxedSignal(inout %arg0 : i64, inout %arg1 : i64, inout %arg2 : i1) {
  llhd.process {
    cf.br ^bb1
  ^bb1:
    %cond = llhd.prb %arg2 : !hw.inout<i1>
    %sig = comb.mux %cond, %arg0, %arg1 : !hw.inout<i64>
    %0 = llhd.prb %sig : !hw.inout<i64>
    // expected-error @+1 {{during process-lowering: the wait terminator is required to have values used in the process as arguments}}
    llhd.wait (%cond : i1), ^bb1
  }
}
