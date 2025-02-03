// RUN: circt-opt --lower-verif-to-sv %s | FileCheck %s

// CHECK-LABEL: hw.module @HasBeenResetAsync
hw.module @HasBeenResetAsync(in %clock: i1, in %reset: i1, out out: i1) {
  %0 = verif.has_been_reset %clock, async %reset
  hw.output %0 : i1

  // CHECK:      %hasBeenResetReg = sv.reg : !hw.inout<i1>

  // CHECK-NEXT: sv.initial {
  // CHECK-NEXT:   sv.if %reset {
  // CHECK-NEXT:     sv.bpassign %hasBeenResetReg, %true : i1
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     sv.bpassign %hasBeenResetReg, %x_i1 : i1
  // CHECK-NEXT:   }
  // CHECK-NEXT: }

  // CHECK-NEXT: sv.always posedge %reset {
  // CHECK-NEXT:   sv.passign %hasBeenResetReg, %true : i1
  // CHECK-NEXT: }

  // CHECK-NEXT: [[REG:%.+]] = sv.read_inout %hasBeenResetReg
  // CHECK-NEXT: [[REG_HI:%.+]] = comb.icmp ceq [[REG]], %true
  // CHECK-NEXT: [[RESET_LO:%.+]] = comb.icmp ceq %reset, %false
  // CHECK-NEXT: [[DONE:%.+]] = comb.and [[REG_HI]], [[RESET_LO]]
  // CHECK-NEXT: %hasBeenReset = hw.wire [[DONE]]

  // CHECK-NEXT: hw.output %hasBeenReset
}

// CHECK-LABEL: hw.module @HasBeenResetSync
hw.module @HasBeenResetSync(in %clock: i1, in %reset: i1, out out: i1) {
  %0 = verif.has_been_reset %clock, sync %reset
  hw.output %0 : i1

  // CHECK:      %hasBeenResetReg = sv.reg : !hw.inout<i1>

  // CHECK-NEXT: sv.initial {
  // CHECK-NEXT:   sv.bpassign %hasBeenResetReg, %x_i1 : i1
  // CHECK-NEXT: }

  // CHECK-NEXT: sv.always posedge %clock {
  // CHECK-NEXT:   sv.if %reset {
  // CHECK-NEXT:     sv.passign %hasBeenResetReg, %true : i1
  // CHECK-NEXT:   }
  // CHECK-NEXT: }

  // CHECK-NEXT: [[REG:%.+]] = sv.read_inout %hasBeenResetReg
  // CHECK-NEXT: [[REG_HI:%.+]] = comb.icmp ceq [[REG]], %true
  // CHECK-NEXT: [[RESET_LO:%.+]] = comb.icmp ceq %reset, %false
  // CHECK-NEXT: [[DONE:%.+]] = comb.and [[REG_HI]], [[RESET_LO]]
  // CHECK-NEXT: %hasBeenReset = hw.wire [[DONE]]

  // CHECK-NEXT: hw.output %hasBeenReset
}
