// RUN: circt-opt --llhd-sig2reg %s | FileCheck %s

func.func private @unknown_time() -> !llhd.time
func.func private @use_signal(%arg0: !llhd.ref<i42>)

// CHECK-LABEL: @RemoveUnprobed
hw.module @RemoveUnprobed(in %d: i42, in %e: i1) {
  %0 = hw.constant 0 : i42
  // CHECK-NOT: llhd.sig
  %a = llhd.sig %0 : <i42>
  %b = llhd.sig %0 : <i42>
  // CHECK-NOT: llhd.drv
  llhd.assign %a, %d : i42
  llhd.assign %b, %d if %e : i42
  // CHECK: hw.output
}

// CHECK-LABEL: hw.module @RemoveUnprobedMultiplyDriven
hw.module @RemoveUnprobedMultiplyDriven(in %a : i42, in %b : i42) {
  // CHECK-NOT: llhd.sig
  %x = llhd.sig %a : <i42>
  %t = llhd.constant_time <0ns, 1d>
  llhd.assign %x, %a : i42
  llhd.drv %x, %b after %t : i42
  // CHECK: hw.output
}

// CHECK-LABEL: hw.module @SkipDrivenFromDifferentBlock
hw.module @SkipDrivenFromDifferentBlock(in %a : i42, in %c : i1, out o : i42) {
  // CHECK: [[SIG:%.+]] = llhd.sig
  %x = llhd.sig %a : <i42>
  scf.if %c {
    // CHECK: llhd.assign [[SIG]], %a
    llhd.assign %x, %a : i42
  }
  // CHECK: [[PRB:%.+]] = llhd.prb [[SIG]]
  %p = llhd.prb %x : i42
  // CHECK: hw.output [[PRB]]
  hw.output %p : i42
}

// CHECK-LABEL: hw.module @SkipUnsupportedUser
hw.module @SkipUnsupportedUser(in %a : i42, out o : i42) {
  // CHECK: [[SIG:%.+]] = llhd.sig
  %x = llhd.sig %a : <i42>
  // CHECK: llhd.assign [[SIG]], %a
  llhd.assign %x, %a : i42
  // CHECK: func.call @use_signal([[SIG]])
  func.call @use_signal(%x) : (!llhd.ref<i42>) -> ()
  // CHECK: [[PRB:%.+]] = llhd.prb [[SIG]]
  %p = llhd.prb %x : i42
  // CHECK: hw.output [[PRB]]
  hw.output %p : i42
}

// CHECK-LABEL: hw.module @SkipMultipleDrivers
hw.module @SkipMultipleDrivers(in %a : i42, in %b : i42, out o : i42) {
  // CHECK: [[SIG:%.+]] = llhd.sig
  %x = llhd.sig %a : <i42>
  %t = llhd.constant_time <0ns, 1d>
  // CHECK: llhd.assign [[SIG]], %a
  llhd.assign %x, %a : i42
  // CHECK: llhd.drv [[SIG]], %b
  llhd.drv %x, %b after %t : i42
  // CHECK: [[PRB:%.+]] = llhd.prb [[SIG]]
  %p = llhd.prb %x : i42
  // CHECK: hw.output [[PRB]]
  hw.output %p : i42
}

// CHECK-LABEL: hw.module @SkipConditionalDrive
hw.module @SkipConditionalDrive(in %a : i42, in %c : i1, out o : i42) {
  // CHECK: [[SIG:%.+]] = llhd.sig
  %x = llhd.sig %a : <i42>
  // CHECK: llhd.assign [[SIG]], %a if %c
  llhd.assign %x, %a if %c : i42
  // CHECK: [[PRB:%.+]] = llhd.prb [[SIG]]
  %p = llhd.prb %x : i42
  // CHECK: hw.output [[PRB]]
  hw.output %p : i42
}

// CHECK-LABEL: hw.module @SkipDelayedDriveWithInit
hw.module @SkipDelayedDriveWithInit(in %a : i42, in %b : i42, out o : i42) {
  // CHECK: [[SIG:%.+]] = llhd.sig
  %x = llhd.sig %a : <i42>
  %t = llhd.constant_time <1ns, 0d>
  // CHECK: llhd.drv [[SIG]], %b
  llhd.drv %x, %b after %t : i42
  // CHECK: [[PRB:%.+]] = llhd.prb [[SIG]]
  %p = llhd.prb %x : i42
  // CHECK: hw.output [[PRB]]
  hw.output %p : i42
}

// CHECK-LABEL: hw.module @SkipNonConstantDriveDelay
hw.module @SkipNonConstantDriveDelay(in %b : i42, out o : i42) {
  // CHECK: [[SIG:%.+]] = llhd.sig :
  %x = llhd.sig : <i42>
  %t = func.call @unknown_time() : () -> !llhd.time
  // CHECK: llhd.drv [[SIG]], %b
  llhd.drv %x, %b after %t : i42
  // CHECK: [[PRB:%.+]] = llhd.prb [[SIG]]
  %p = llhd.prb %x : i42
  // CHECK: hw.output [[PRB]]
  hw.output %p : i42
}

// CHECK-LABEL: hw.module @SkipUndrivenWithoutInit
hw.module @SkipUndrivenWithoutInit(out o : i42) {
  // CHECK: [[SIG:%.+]] = llhd.sig :
  %x = llhd.sig : <i42>
  // CHECK: [[PRB:%.+]] = llhd.prb [[SIG]]
  %p = llhd.prb %x : i42
  // CHECK: hw.output [[PRB]]
  hw.output %p : i42
}

// CHECK-LABEL: hw.module @SkipUndrivenWithNonConstantInit
hw.module @SkipUndrivenWithNonConstantInit(in %a : i42, out o : i42) {
  // CHECK: [[SIG:%.+]] = llhd.sig %a
  %x = llhd.sig %a : <i42>
  // CHECK: [[PRB:%.+]] = llhd.prb [[SIG]]
  %p = llhd.prb %x : i42
  // CHECK: hw.output [[PRB]]
  hw.output %p : i42
}

// CHECK-LABEL: hw.module @PromoteUndrivenWithConstantInit
hw.module @PromoteUndrivenWithConstantInit(out o : i42) {
  // CHECK-NOT: llhd.sig
  // CHECK: [[C:%.+]] = hw.constant 9 : i42
  %c = hw.constant 9 : i42
  %x = llhd.sig %c : <i42>
  %p = llhd.prb %x : i42
  // CHECK: hw.output [[C]]
  hw.output %p : i42
}

// CHECK-LABEL: hw.module @PromoteAssign
hw.module @PromoteAssign(in %a : i42, in %b : i42, out o1 : i42, out o2 : i42) {
  // CHECK-NOT: llhd.sig
  %x = llhd.sig %a : <i42>
  llhd.assign %x, %b : i42
  %p1 = llhd.prb %x : i42
  %p2 = llhd.prb %x : i42
  // CHECK: hw.output %b, %b
  hw.output %p1, %p2 : i42, i42
}

// CHECK-LABEL: hw.module @PromoteDelayedDrive
hw.module @PromoteDelayedDrive(in %b : i42, out o : i42) {
  // CHECK-NOT: llhd.sig
  // CHECK-NOT: llhd.drv
  %x = llhd.sig : <i42>
  %t = llhd.constant_time <1ns, 0d>
  llhd.drv %x, %b after %t : i42
  %p = llhd.prb %x : i42
  // CHECK: [[DLY:%.+]] = llhd.delay %b by <1ns, 0d> : i42
  // CHECK: hw.output [[DLY]]
  hw.output %p : i42
}
