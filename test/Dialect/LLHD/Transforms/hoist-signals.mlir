// RUN: circt-opt --llhd-hoist-signals %s | FileCheck %s

// CHECK-LABEL: @SimpleProbes
hw.module @SimpleProbes() {
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

// CHECK-LABEL: @DontHoistProbesAcrossSideEffects
hw.module @DontHoistProbesAcrossSideEffects() {
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

// CHECK-LABEL: @DontHoistProbesIfLeakingAcrossWait
hw.module @DontHoistProbesIfLeakingAcrossWait() {
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
    // CHECK-NEXT: llhd.drv %a
    llhd.prb %a : !hw.inout<i42>
    llhd.drv %a, %c0_i42 after %0 : !hw.inout<i42>
    llhd.halt
  }
}

// CHECK-LABEL: @SimpleDrives
hw.module @SimpleDrives(in %u: i42, in %v: i42, in %w: i42) {
  // CHECK: [[DEL:%.+]] = llhd.constant_time <0ns, 1d, 0e>
  // CHECK: [[EPS:%.+]] = llhd.constant_time <0ns, 0d, 1e>
  %0 = llhd.constant_time <0ns, 1d, 0e>
  %1 = llhd.constant_time <0ns, 0d, 1e>
  %c0_i42 = hw.constant 0 : i42
  %a = llhd.sig %c0_i42 : i42
  %b = llhd.sig %c0_i42 : i42
  %c = llhd.sig %c0_i42 : i42
  %d = llhd.sig %c0_i42 : i42
  // CHECK: [[RES:%.+]]:7 = llhd.process -> i42, i42, i42, i42, !llhd.time, i42, i1 {
  %2 = llhd.process -> i42 {
    // CHECK-NEXT: cf.br ^bb1
    cf.br ^bb1
  ^bb1:
    // CHECK-NEXT: ^bb1:
    // CHECK-NOT: llhd.drv
    llhd.drv %a, %u after %0 : !hw.inout<i42>
    llhd.drv %b, %u after %0 : !hw.inout<i42>
    llhd.drv %c, %u after %0 : !hw.inout<i42>
    llhd.drv %d, %u after %0 : !hw.inout<i42>
    // CHECK-NEXT: llhd.wait yield (%w, %u, %u, %u, [[DEL]], %u, %true : i42, i42, i42, i42, !llhd.time, i42, i1), ^bb2
    llhd.wait yield (%w : i42), ^bb2
  ^bb2:
    // CHECK-NEXT: ^bb2:
    // CHECK-NEXT: cf.br ^bb3
    cf.br ^bb3
  ^bb3:
    // CHECK-NEXT: ^bb3:
    // CHECK-NOT: llhd.drv
    llhd.drv %a, %u after %0 : !hw.inout<i42>
    llhd.drv %b, %v after %0 : !hw.inout<i42>
    llhd.drv %c, %u after %1 : !hw.inout<i42>
    // CHECK-NEXT: llhd.halt %w, %u, %v, %u, [[EPS]], {{%c0_i42.*}}, %false : i42, i42, i42, i42, !llhd.time, i42, i1
    llhd.halt %w : i42
  }
  // CHECK-NEXT: }
  // CHECK-NEXT: llhd.drv %a, [[RES]]#1 after [[DEL]] :
  // CHECK-NEXT: llhd.drv %b, [[RES]]#2 after [[DEL]] :
  // CHECK-NEXT: llhd.drv %c, [[RES]]#3 after [[RES]]#4 :
  // CHECK-NEXT: llhd.drv %d, [[RES]]#5 after [[DEL]] if [[RES]]#6 :
  // CHECK-NEXT: call @use_i42([[RES]]#0)
  func.call @use_i42(%2) : (i42) -> ()
}

// CHECK-LABEL: @DontHoistDrivesAcrossSideEffects
hw.module @DontHoistDrivesAcrossSideEffects() {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %c0_i42 = hw.constant 0 : i42
  %a = llhd.sig %c0_i42 : i42
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NEXT: llhd.drv %a
    llhd.drv %a, %c0_i42 after %0 : !hw.inout<i42>
    // CHECK-NEXT: call @maybe_side_effecting
    func.call @maybe_side_effecting() : () -> ()
    llhd.halt
  }
}

// CHECK-LABEL: @DontHoistDrivesOfSlotsWithUnsavoryUsers
hw.module @DontHoistDrivesOfSlotsWithUnsavoryUsers(inout %c: i42) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %c0_i42 = hw.constant 0 : i42
  %a = llhd.sig %c0_i42 : i42
  %b = llhd.sig %c0_i42 : i42
  // Unsavory uses outside of process don't affect hoistability.
  // CHECK: call @use_inout_i42(%a)
  func.call @use_inout_i42(%a) : (!hw.inout<i42>) -> ()
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NOT: llhd.drv
    llhd.drv %a, %c0_i42 after %0 : !hw.inout<i42>
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NEXT: call @use_inout_i42(%b)
    func.call @use_inout_i42(%b) : (!hw.inout<i42>) -> ()
    // CHECK-NEXT: llhd.drv %b
    llhd.drv %b, %c0_i42 after %0 : !hw.inout<i42>
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NEXT: llhd.drv %c
    llhd.drv %c, %c0_i42 after %0 : !hw.inout<i42>
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
}

// CHECK-LABEL: @OnlyHoistLastDrive
hw.module @OnlyHoistLastDrive(in %u: i42, in %v: i42) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %c0_i42 = hw.constant 0 : i42
  %a = llhd.sig %c0_i42 : i42
  %b = llhd.sig %c0_i42 : i42
  // CHECK: [[RES:%.+]]:2 = llhd.process -> i42, i42
  llhd.process {
    // CHECK-NEXT: llhd.drv %a, %u
    llhd.drv %a, %u after %0 : !hw.inout<i42>
    // CHECK-NEXT: llhd.drv %b, %v
    llhd.drv %b, %v after %0 : !hw.inout<i42>
    // CHECK-NOT: llhd.drv %a, %v
    llhd.drv %a, %v after %0 : !hw.inout<i42>
    // CHECK-NOT: llhd.drv %b, %u
    llhd.drv %b, %u after %0 : !hw.inout<i42>
    // CHECK-NEXT: llhd.halt %v, %u : i42, i42
    llhd.halt
  }
  // CHECK: llhd.drv %a, [[RES]]#0
  // CHECK: llhd.drv %b, [[RES]]#1
}

func.func private @use_i42(%arg0: i42)
func.func private @use_inout_i42(%arg0: !hw.inout<i42>)
func.func private @maybe_side_effecting()
