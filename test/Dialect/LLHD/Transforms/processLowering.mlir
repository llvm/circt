// RUN: circt-opt %s -llhd-process-lowering -split-input-file -verify-diagnostics | FileCheck %s

// check that input and output signals are transferred correctly
// CHECK-LABEL: hw.module @inputAndOutput
hw.module @inputAndOutput(inout %arg0 : i64, inout %arg1 : i1, inout %arg2 : i1) {
  llhd.process {
    // CHECK-NEXT: hw.output
    llhd.halt
  }
}

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
