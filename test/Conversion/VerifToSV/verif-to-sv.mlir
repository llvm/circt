// RUN: circt-opt --lower-verif-to-sv %s | FileCheck %s

// CHECK: sv.macro.decl @SYNTHESIS

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

  // CHECK-NEXT: sv.always posedge %clock, posedge %reset {
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

// CHECK-LABEL: hw.module @AssertLike
hw.module @AssertLike(in %clock: i1, in %p: i1, in %en: i1) {
  // CHECK: [[T:%.+]] = hw.constant true
  // CHECK: [[D:%.+]] = comb.xor %en, [[T]]
  // CHECK: sv.ifdef @SYNTHESIS {
  // CHECK: } else {
  // CHECK:   sv.assert_property %p disable_iff [[D]] label "a" : i1
  // CHECK: }
  verif.assert %p if %en label "a" : i1
  // CHECK: sv.ifdef @SYNTHESIS {
  // CHECK: } else {
  // CHECK:   sv.assume_property %p label "u" : i1
  // CHECK: }
  verif.assume %p label "u" : i1
  // CHECK: sv.ifdef @SYNTHESIS {
  // CHECK: } else {
  // CHECK:   sv.cover_property %p label "c" : i1
  // CHECK: }
  verif.cover %p label "c" : i1
}

// CHECK-LABEL: hw.module @ClockedAssertLike
hw.module @ClockedAssertLike(in %clock: i1, in %p: i1) {
  // CHECK: sv.ifdef @SYNTHESIS {
  // CHECK: } else {
  // CHECK:   sv.assert_property %p on posedge %clock label "a" : i1
  // CHECK: }
  verif.clocked_assert %p, posedge %clock label "a" : i1
  // CHECK: sv.ifdef @SYNTHESIS {
  // CHECK: } else {
  // CHECK:   sv.assume_property %p on negedge %clock label "u" : i1
  // CHECK: }
  verif.clocked_assume %p, negedge %clock label "u" : i1
  // CHECK: sv.ifdef @SYNTHESIS {
  // CHECK: } else {
  // CHECK:   sv.cover_property %p on edge %clock label "c" : i1
  // CHECK: }
  verif.clocked_cover %p, edge %clock label "c" : i1
}
