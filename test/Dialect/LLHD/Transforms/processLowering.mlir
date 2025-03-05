// RUN: circt-opt %s -llhd-process-lowering | FileCheck %s

// check wait suspended process
// CHECK-LABEL: hw.module @simpleWait
hw.module @simpleWait() {
  llhd.process {
    // CHECK-NEXT: hw.output
    cf.br ^bb1
  ^bb1:
    llhd.wait ^bb1
  }
}

// Check wait with observing probed signals
// CHECK-LABEL: hw.module @prbAndWait
hw.module @prbAndWait(inout %arg0 : i64) {
  // CHECK-NEXT: %{{.*}} = llhd.prb
  %1 = llhd.prb %arg0 : !hw.inout<i64>
  llhd.process {
    // CHECK-NEXT: %{{.*}} = llhd.prb
    // CHECK-NEXT: hw.output
    cf.br ^bb1
  ^bb1:
    %0 = llhd.prb %arg0 : !hw.inout<i64>
    llhd.wait (%1 : i64), ^bb1
  }
}

// Check wait with observing probed signals
// CHECK-LABEL: hw.module @prbAndWaitMoreObserved
hw.module @prbAndWaitMoreObserved(inout %arg0 : i64, inout %arg1 : i64) {
  // CHECK-NEXT: %{{.*}} = llhd.prb
  %1 = llhd.prb %arg0 : !hw.inout<i64>
  // CHECK-NEXT: %{{.*}} = llhd.prb
  %2 = llhd.prb %arg1 : !hw.inout<i64>
  llhd.process {
    // CHECK-NEXT: %{{.*}} = llhd.prb
    // CHECK-NEXT: hw.output
    cf.br ^bb1
  ^bb1:
    %0 = llhd.prb %arg0 : !hw.inout<i64>
    llhd.wait (%1, %2 : i64, i64), ^bb1
  }
}

// CHECK-LABEL: hw.module @muxedSignal
hw.module @muxedSignal(inout %arg0 : i64, inout %arg1 : i64) {
  // CHECK-NEXT: %{{.*}} = llhd.prb
  %1 = llhd.prb %arg0 : !hw.inout<i64>
  // CHECK-NEXT: %{{.*}} = llhd.prb
  %2 = llhd.prb %arg1 : !hw.inout<i64>
  llhd.process {
    cf.br ^bb1
  ^bb1:
    // CHECK-NEXT: %{{.*}} = hw.constant
    // CHECK-NEXT: %{{.*}} = comb.mux
    // CHECK-NEXT: %{{.*}} = llhd.prb
    // CHECK-NEXT: hw.output
    %cond = hw.constant true
    %sig = comb.mux %cond, %arg0, %arg1 : !hw.inout<i64>
    %0 = llhd.prb %sig : !hw.inout<i64>
    llhd.wait (%1, %2 : i64, i64), ^bb1
  }
}

// CHECK-LABEL: hw.module @muxedSignal2
hw.module @muxedSignal2(inout %arg0 : i64, inout %arg1 : i64) {
  // CHECK-NEXT: %{{.*}} = hw.constant
  // CHECK-NEXT: %{{.*}} = comb.mux
  // CHECK-NEXT: %{{.*}} = llhd.prb
  %cond = hw.constant true
  %sig = comb.mux %cond, %arg0, %arg1 : !hw.inout<i64>
  %0 = llhd.prb %sig : !hw.inout<i64>
  llhd.process {
    cf.br ^bb1
  ^bb1:
    // CHECK-NEXT: comb.and
    %1 = comb.and %0, %0 : i64
    // CHECK-NEXT: hw.output
    llhd.wait (%0 : i64), ^bb1
  }
}

// CHECK-LABEL: hw.module @partialSignal
hw.module @partialSignal(inout %arg0 : i64) {
  // CHECK-NEXT: %{{.*}} = llhd.prb
  %1 = llhd.prb %arg0 : !hw.inout<i64>
  llhd.process {
    cf.br ^bb1
  ^bb1:
    // CHECK-NEXT: %{{.*}} = hw.constant
    // CHECK-NEXT: %{{.*}} = llhd.sig.extract
    // CHECK-NEXT: %{{.*}} = llhd.prb
    // CHECK-NEXT: hw.output
    %c = hw.constant 16 : i6
    %sig = llhd.sig.extract %arg0 from %c : (!hw.inout<i64>) -> !hw.inout<i32>
    %0 = llhd.prb %sig : !hw.inout<i32>
    llhd.wait (%1 : i64), ^bb1
  }
}

// COM: Tests to check that non-combinational processes are not inlined.

// Check wait with observing probed signals
// CHECK-LABEL: @prbAndWaitNotObserved
// CHECK: llhd.process
hw.module @prbAndWaitNotObserved(inout %arg0 : i64) {
  llhd.process {
    cf.br ^bb1
  ^bb1:
    %0 = llhd.prb %arg0 : !hw.inout<i64>
    llhd.wait ^bb1
  }
}

// Check that block arguments for the second block are not allowed.
// CHECK-LABEL: @blockArgumentsNotAllowed
// CHECK: llhd.process
hw.module @blockArgumentsNotAllowed(inout %arg0 : i64) {
  llhd.process {
    %prb = llhd.prb %arg0 : !hw.inout<i64>
    cf.br ^bb1(%prb : i64)
  ^bb1(%a : i64):
    llhd.wait ^bb1(%a: i64)
  }
}

// Check that the entry block is terminated by a cf.br terminator.
// CHECK-LABEL: @entryBlockMustHaveBrTerminator
// CHECK: llhd.process
hw.module @entryBlockMustHaveBrTerminator() {
  llhd.process {
    llhd.wait ^bb1
  ^bb1:
    llhd.wait ^bb1
  }
}

// Check that there is no optional time operand in the wait terminator.
// CHECK-LABEL: @noOptionalTime
// CHECK: llhd.process
hw.module @noOptionalTime() {
  llhd.process {
    cf.br ^bb1
  ^bb1:
    %time = llhd.constant_time #llhd.time<0ns, 0d, 0e>
    llhd.wait delay %time, ^bb1
  }
}

// Check that if there are two blocks, the second one is terminated by a wait terminator.
// CHECK-LABEL: @secondBlockTerminatedByWait
// CHECK: llhd.process
hw.module @secondBlockTerminatedByWait() {
  llhd.process {
    cf.br ^bb1
  ^bb1:
    llhd.halt
  }
}

// Check that there are not more than two blocks.
// CHECK-LABEL: @moreThanTwoBlocksNotAllowed
// CHECK: llhd.process
hw.module @moreThanTwoBlocksNotAllowed() {
  llhd.process {
    cf.br ^bb1
  ^bb1:
    cf.br ^bb2
  ^bb2:
    llhd.wait ^bb1
  }
}

// CHECK-LABEL: @muxedSignalNotLowerable
// CHECK: llhd.process
hw.module @muxedSignalNotLowerable(inout %arg0 : i64, inout %arg1 : i64, inout %arg2 : i1) {
  llhd.process {
    cf.br ^bb1
  ^bb1:
    %cond = llhd.prb %arg2 : !hw.inout<i1>
    %sig = comb.mux %cond, %arg0, %arg1 : !hw.inout<i64>
    %0 = llhd.prb %sig : !hw.inout<i64>
    llhd.wait (%cond : i1), ^bb1
  }
}
