// RUN: circt-opt %s -cse | FileCheck %s

// CHECK-LABEL: @checkPrbDceAndCseIn
hw.module @checkPrbDceAndCseIn(inout %arg0 : i32, inout %arg1 : i32, inout %arg2 : i32) {
  // CHECK-NEXT: llhd.constant_time
  %time = llhd.constant_time <0ns, 1d, 0e>

  // CHECK-NEXT: [[P0:%.*]] = llhd.prb
  %1 = llhd.prb %arg0 : !hw.inout<i32>
  %2 = llhd.prb %arg0 : !hw.inout<i32>
  %3 = llhd.prb %arg0 : !hw.inout<i32>

  // CHECK-NEXT: llhd.drv %arg1, [[P0]]
  // CHECK-NEXT: llhd.drv %arg2, [[P0]]
  llhd.drv %arg1, %1 after %time : !hw.inout<i32>
  llhd.drv %arg2, %2 after %time : !hw.inout<i32>
}

// CHECK-LABEL: @checkPrbDceButNotCse
hw.module @checkPrbDceButNotCse(inout %arg0 : i32, inout %arg1 : i32, inout %arg2 : i32) {
  %prb = llhd.prb %arg0 : !hw.inout<i32>
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NEXT: llhd.constant_time
    %time = llhd.constant_time <0ns, 1d, 0e>

    // CHECK-NEXT: [[P1:%.*]] = llhd.prb
    %1 = llhd.prb %arg0 : !hw.inout<i32>
    // CHECK-NEXT: llhd.wait
    llhd.wait (%prb: i32), ^bb1
  // CHECK-NEXT: ^bb1:
  ^bb1:
    // CHECK-NEXT: [[P2:%.*]] = llhd.prb
    %2 = llhd.prb %arg0 : !hw.inout<i32>
    %3 = llhd.prb %arg0 : !hw.inout<i32>

    // CHECK-NEXT: llhd.drv %arg1, [[P1]]
    // CHECK-NEXT: llhd.drv %arg2, [[P2]]
    llhd.drv %arg1, %1 after %time : !hw.inout<i32>
    llhd.drv %arg2, %2 after %time : !hw.inout<i32>
    llhd.halt
  }
  hw.output
}
