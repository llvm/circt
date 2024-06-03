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
  // CHECK-NEXT: [[DONE:%.+]] = comb.and bin [[REG_HI]], [[RESET_LO]]
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
  // CHECK-NEXT: [[DONE:%.+]] = comb.and bin [[REG_HI]], [[RESET_LO]]
  // CHECK-NEXT: %hasBeenReset = hw.wire [[DONE]]

  // CHECK-NEXT: hw.output %hasBeenReset
}

// CHECK-LABEL: hw.module @ClockedAsserts
hw.module @ClockedAsserts(in %clock: i1, in %reset: i1, %a : i1, %b : i1, out out: i1) {
  %0 = ltl.not %a : i1
  %1 = ltl.or %0, %b : !ltl.property, i1
  %one = hw.constant 1 : i1
  %not_b = comb.xor %b, %one : i1
  %2 = ltl.and %0, %not_b : !ltl.property, i1 

  // CHECK: sv.sva.assert_property posedge %clock disable %b, %0 : !ltl.property
  verif.clocked_assert %1 clock posedge %clock : !ltl.property
  // CHECK: sv.sva.assume_property posedge %clock disable %b, %0 : !ltl.property
  verif.clocked_assume %1 clock posedge %clock : !ltl.property
  // CHECK: sv.sva.cover_property posedge %clock disable %b, %0 : !ltl.property
  verif.clocked_cover %2 clock posedge %clock : !ltl.property
}
