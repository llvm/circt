// RUN: circt-opt %s -llhd-process-lowering -split-input-file -verify-diagnostics | FileCheck %s

// no inputs and outputs
// CHECK-LABEL: hw.module @empty
// CHECK-SAME: ()
llhd.proc @empty() -> () {
  // CHECK-NEXT: hw.output
  llhd.halt
}

// check that input and output signals are transferred correctly
// CHECK-LABEL: hw.module @inputAndOutput
// CHECK-SAME: (inout %{{.*}} : i64, inout %{{.*}} : i1, inout %{{.*}} : i1)
llhd.proc @inputAndOutput(%arg0 : !hw.inout<i64>, %arg1 : !hw.inout<i1>) -> (%arg2 : !hw.inout<i1>) {
  // CHECK-NEXT: hw.output
  llhd.halt
}

// check wait suspended process
// CHECK-LABEL: hw.module @simpleWait
// CHECK-SAME: ()
llhd.proc @simpleWait() -> () {
  // CHECK-NEXT: hw.output
  cf.br ^bb1
^bb1:
  llhd.wait ^bb1
}

// Check wait with observing probed signals
// CHECK-LABEL: hw.module @prbAndWait
// CHECK-SAME: (inout %{{.*}} : i64)
llhd.proc @prbAndWait(%arg0 : !hw.inout<i64>) -> () {
  // CHECK-NEXT: %{{.*}} = llhd.prb
  // CHECK-NEXT: hw.output
  cf.br ^bb1
^bb1:
  %0 = llhd.prb %arg0 : !hw.inout<i64>
  llhd.wait (%arg0 : !hw.inout<i64>), ^bb1
}

// Check wait with observing probed signals
// CHECK-LABEL: hw.module @prbAndWaitMoreObserved
// CHECK-SAME: (inout %{{.*}} : i64, inout %{{.*}} : i64)
llhd.proc @prbAndWaitMoreObserved(%arg0 : !hw.inout<i64>, %arg1 : !hw.inout<i64>) -> () {
  // CHECK-NEXT: %{{.*}} = llhd.prb
  // CHECK-NEXT: hw.output
  cf.br ^bb1
^bb1:
  %0 = llhd.prb %arg0 : !hw.inout<i64>
  llhd.wait (%arg0, %arg1 : !hw.inout<i64>, !hw.inout<i64>), ^bb1
}

// CHECK-LABEL: hw.module @muxedSignal
llhd.proc @muxedSignal(%arg0 : !hw.inout<i64>, %arg1 : !hw.inout<i64>) -> () {
  cf.br ^bb1
^bb1:
  // CHECK-NEXT: %{{.*}} = hw.constant
  // CHECK-NEXT: %{{.*}} = comb.mux
  // CHECK-NEXT: %{{.*}} = llhd.prb
  // CHECK-NEXT: hw.output
  %cond = hw.constant true
  %sig = comb.mux %cond, %arg0, %arg1 : !hw.inout<i64>
  %0 = llhd.prb %sig : !hw.inout<i64>
  llhd.wait (%arg0, %arg1 : !hw.inout<i64>, !hw.inout<i64>), ^bb1
}

// CHECK-LABEL: hw.module @muxedSignal2
llhd.proc @muxedSignal2(%arg0 : !hw.inout<i64>, %arg1 : !hw.inout<i64>) -> () {
  cf.br ^bb1
^bb1:
  // CHECK-NEXT: %{{.*}} = hw.constant
  // CHECK-NEXT: %{{.*}} = comb.mux
  // CHECK-NEXT: %{{.*}} = llhd.prb
  // CHECK-NEXT: hw.output
  %cond = hw.constant true
  %sig = comb.mux %cond, %arg0, %arg1 : !hw.inout<i64>
  %0 = llhd.prb %sig : !hw.inout<i64>
  llhd.wait (%sig : !hw.inout<i64>), ^bb1
}

// CHECK-LABEL: hw.module @partialSignal
llhd.proc @partialSignal(%arg0 : !hw.inout<i64>) -> () {
  cf.br ^bb1
^bb1:
  // CHECK-NEXT: %{{.*}} = hw.constant
  // CHECK-NEXT: %{{.*}} = llhd.sig.extract
  // CHECK-NEXT: %{{.*}} = llhd.prb
  // CHECK-NEXT: hw.output
  %c = hw.constant 16 : i6
  %sig = llhd.sig.extract %arg0 from %c : (!hw.inout<i64>) -> !hw.inout<i32>
  %0 = llhd.prb %sig : !hw.inout<i32>
  llhd.wait (%arg0 : !hw.inout<i64>), ^bb1
}
