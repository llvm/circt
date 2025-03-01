// RUN: circt-opt --llhd-hoist-signals %s | FileCheck %s

// CHECK-LABEL: @Simple
hw.module @Simple() {
  %c0_i42 = hw.constant 0 : i42
  %a = llhd.sig %c0_i42 : i42
  // CHECK: llhd.sig
  // CHECK-NEXT: [[A:%.+]] = llhd.prb %a
  // CHECK-NEXT: llhd.process
  llhd.process {
    // CHECK-NOT: llhd.prb
    %0 = llhd.prb %a : !hw.inout<i42>
    // CHECK-NEXT: call @use_i42([[A]])
    func.call @use_i42(%0) : (i42) -> ()
    llhd.wait ^bb1
  ^bb1:
    // CHECK: ^bb1:
    // CHECK-NOT: llhd.prb
    %1 = llhd.prb %a : !hw.inout<i42>
    // CHECK-NEXT: call @use_i42([[A]])
    func.call @use_i42(%1) : (i42) -> ()
    llhd.halt
  }
}

// CHECK-LABEL: @DontHoistAcrossSideEffects
hw.module @DontHoistAcrossSideEffects() {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %c0_i42 = hw.constant 0 : i42
  %a = llhd.sig %c0_i42 : i42
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NEXT: llhd.drv
    llhd.drv %a, %c0_i42 after %0 : !hw.inout<i42>
    // CHECK-NEXT: llhd.prb %a
    llhd.prb %a : !hw.inout<i42>
    cf.br ^bb1
  ^bb1:
    // CHECK: ^bb1:
    // CHECK-NEXT: llhd.prb %a
    llhd.prb %a : !hw.inout<i42>
    llhd.wait ^bb2
  ^bb2:
    // CHECK: ^bb2:
    // CHECK-NEXT: call @maybe_side_effecting
    func.call @maybe_side_effecting() : () -> ()
    // CHECK-NEXT: llhd.prb %a
    llhd.prb %a : !hw.inout<i42>
    llhd.halt
  }
}

// CHECK-LABEL: @DontHoistIfLeakingAcrossWait
hw.module @DontHoistIfLeakingAcrossWait() {
  %c0_i42 = hw.constant 0 : i42
  %a = llhd.sig %c0_i42 : i42
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NEXT: [[A:%.+]] = llhd.prb %a
    %0 = llhd.prb %a : !hw.inout<i42>
    llhd.wait ^bb1
  ^bb1:
    // CHECK: ^bb1:
    // CHECK-NEXT: call @use_i42([[A]])
    func.call @use_i42(%0) : (i42) -> ()
    llhd.halt
  }
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NEXT: llhd.prb %a
    %0 = llhd.prb %a : !hw.inout<i42>
    cf.br ^bb1(%0 : i42)
  ^bb1(%1: i42):
    // CHECK: ^bb1([[A:%.+]]: i42):
    llhd.wait ^bb2
  ^bb2:
    // CHECK: ^bb2:
    // CHECK-NEXT: call @use_i42([[A]])
    func.call @use_i42(%1) : (i42) -> ()
    llhd.halt
  }
}

// CHECK-LABEL: @DontHoistLocalSignals
hw.module @DontHoistLocalSignals() {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %c0_i42 = hw.constant 0 : i42
  // CHECK: llhd.process
  llhd.process {
    %a = llhd.sig %c0_i42 : i42
    llhd.wait ^bb1
  ^bb1:
    // CHECK: ^bb1:
    // CHECK-NEXT: llhd.prb %a
    llhd.prb %a : !hw.inout<i42>
    llhd.halt
  }
}

func.func private @use_i42(%arg0: i42)
func.func private @maybe_side_effecting()
