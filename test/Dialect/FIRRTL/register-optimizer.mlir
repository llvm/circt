// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-register-optimizer)))' %s | FileCheck %s

firrtl.circuit "invalidReg"   {
  // CHECK-LABEL: @invalidReg
  firrtl.module @invalidReg(in %clock: !firrtl.clock, out %a: !firrtl.uint<1>) {
    %foobar = firrtl.reg %clock  : !firrtl.clock, !firrtl.uint<1>
    firrtl.matchingconnect %foobar, %foobar : !firrtl.uint<1>
    //CHECK-NOT: firrtl.connect %foobar, %foobar
    //CHECK: %[[inv:.*]] = firrtl.invalidvalue
    //CHECK: firrtl.matchingconnect %a, %[[inv]]
    firrtl.matchingconnect %a, %foobar : !firrtl.uint<1>
  }

  // Initial-value folding tests.
  // CHECK-LABEL: @selfDrivenRegInitial
  firrtl.module @selfDrivenRegInitial(in %clock: !firrtl.clock, out %one: !firrtl.uint<1>, out %zero: !firrtl.uint<1>) {
    %regOne = firrtl.reg %clock {initial = 1 : ui1} : !firrtl.clock, !firrtl.uint<1>
    %regZero = firrtl.reg %clock {initial = 0 : ui1} : !firrtl.clock, !firrtl.uint<1>
    firrtl.matchingconnect %regOne, %regOne : !firrtl.uint<1>
    firrtl.matchingconnect %regZero, %regZero : !firrtl.uint<1>
    // CHECK-NOT: firrtl.invalidvalue
    // CHECK: %[[one:.*]] = firrtl.constant 1
    // CHECK: %[[zero:.*]] = firrtl.constant 0
    // CHECK: firrtl.matchingconnect %one, %[[one]]
    // CHECK: firrtl.matchingconnect %zero, %[[zero]]
    firrtl.matchingconnect %one, %regOne : !firrtl.uint<1>
    firrtl.matchingconnect %zero, %regZero : !firrtl.uint<1>
  }

  // CHECK-LABEL: @constantRegWriteInitial
  firrtl.module @constantRegWriteInitial(in %clock: !firrtl.clock, out %mismatch: !firrtl.uint<1>, out %match: !firrtl.uint<1>) {
    %c = firrtl.constant 0 : !firrtl.uint<1>
    %mismatchReg = firrtl.reg %clock {initial = 1 : ui1} : !firrtl.clock, !firrtl.uint<1>
    firrtl.matchingconnect %mismatchReg, %c : !firrtl.uint<1>
    %matchReg = firrtl.reg %clock {initial = 0 : ui1} : !firrtl.clock, !firrtl.uint<1>
    firrtl.matchingconnect %matchReg, %c : !firrtl.uint<1>
    // CHECK: %[[reg:.*]] = firrtl.reg %clock {initial = 1 : ui1}
    // CHECK: firrtl.matchingconnect %[[reg]], %c
    // CHECK-NOT: firrtl.reg
    // CHECK: firrtl.matchingconnect %mismatch, %[[reg]]
    // CHECK: firrtl.matchingconnect %match, %c
    firrtl.matchingconnect %mismatch, %mismatchReg : !firrtl.uint<1>
    firrtl.matchingconnect %match, %matchReg : !firrtl.uint<1>
  }

  // CHECK-LABEL: @constantRegResetWriteInitial
  firrtl.module @constantRegResetWriteInitial(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, out %mismatch: !firrtl.uint<1>, out %match: !firrtl.uint<1>) {
    %c = firrtl.constant 0 : !firrtl.uint<1>
    %mismatchReg = firrtl.regreset %clock, %reset, %c {initial = 1 : ui1} : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.matchingconnect %mismatchReg, %c : !firrtl.uint<1>
    %matchReg = firrtl.regreset %clock, %reset, %c {initial = 0 : ui1} : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.matchingconnect %matchReg, %c : !firrtl.uint<1>
    // CHECK: %[[reg:.*]] = firrtl.regreset {{.*}} {initial = 1 : ui1}
    // CHECK: firrtl.matchingconnect %[[reg]], %c
    // CHECK-NOT: firrtl.regreset
    // CHECK: firrtl.matchingconnect %mismatch, %[[reg]]
    // CHECK: firrtl.matchingconnect %match, %c
    firrtl.matchingconnect %mismatch, %mismatchReg : !firrtl.uint<1>
    firrtl.matchingconnect %match, %matchReg : !firrtl.uint<1>
  }

  // CHECK-LABEL: @constantRegWrite
  firrtl.module @constantRegWrite(in %clock: !firrtl.clock, out %a: !firrtl.uint<1>) {
    %c = firrtl.constant 0 : !firrtl.uint<1>
    %foobar = firrtl.reg %clock  : !firrtl.clock, !firrtl.uint<1>
    firrtl.matchingconnect %foobar, %c : !firrtl.uint<1>
    //CHECK-NOT: firrtl.connect %foobar, %c
    //CHECK: %[[const:.*]] = firrtl.constant
    //CHECK: firrtl.matchingconnect %a, %[[const]]
    firrtl.matchingconnect %a, %foobar : !firrtl.uint<1>
  }

  // CHECK-LABEL: @constantRegWriteDom
  firrtl.module @constantRegWriteDom(in %clock: !firrtl.clock, out %a: !firrtl.uint<1>) {
    %foobar = firrtl.reg %clock  : !firrtl.clock, !firrtl.uint<1>
    //CHECK-NOT: firrtl.connect %foobar, %c
    //CHECK: %[[const:.*]] = firrtl.constant
    //CHECK: firrtl.matchingconnect %a, %[[const]]
    firrtl.matchingconnect %a, %foobar : !firrtl.uint<1>
    %c = firrtl.constant 0 : !firrtl.uint<1>
    firrtl.matchingconnect %foobar, %c : !firrtl.uint<1>
  }

  // CHECK-LABEL: @constantRegResetWrite
  firrtl.module @constantRegResetWrite(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, out %a: !firrtl.uint<1>) {
    %c = firrtl.constant 0 : !firrtl.uint<1>
    %foobar = firrtl.regreset %clock, %reset, %c  : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.matchingconnect %foobar, %c : !firrtl.uint<1>
    //CHECK-NOT: firrtl.connect %foobar, %c
    //CHECK: %[[const:.*]] = firrtl.constant
    //CHECK: firrtl.matchingconnect %a, %[[const]]
    firrtl.matchingconnect %a, %foobar : !firrtl.uint<1>
  }

  // CHECK-LABEL: @constantRegResetWriteSelf
  firrtl.module @constantRegResetWriteSelf(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, out %a: !firrtl.uint<1>) {
    %c = firrtl.constant 0 : !firrtl.uint<1>
    %foobar = firrtl.regreset %clock, %reset, %c  : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.matchingconnect %foobar, %foobar : !firrtl.uint<1>
    //CHECK-NOT: firrtl.connect %foobar, %c
    //CHECK: %[[const:.*]] = firrtl.constant
    //CHECK: firrtl.matchingconnect %a, %[[const]]
    firrtl.matchingconnect %a, %foobar : !firrtl.uint<1>
  }

  // CHECK-LABEL: @movedFromIMCP
  firrtl.module @movedFromIMCP(
        in %clock: !firrtl.clock,
        in %reset: !firrtl.uint<1>,
        out %result6: !firrtl.uint<2>,
        out %result7: !firrtl.uint<4>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
    %c0_ui4 = firrtl.constant 0 : !firrtl.uint<4>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>

    // regreset
    %regreset = firrtl.regreset %clock, %reset, %c0_ui2 : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<2>

    firrtl.matchingconnect %regreset, %c0_ui2 : !firrtl.uint<2>

    // CHECK: firrtl.matchingconnect %result6, %c0_ui2
    firrtl.matchingconnect %result6, %regreset: !firrtl.uint<2>

    // reg
    %reg = firrtl.reg %clock  : !firrtl.clock, !firrtl.uint<4>
    firrtl.matchingconnect %reg, %c0_ui4 : !firrtl.uint<4>
    // CHECK: firrtl.matchingconnect %result7, %c0_ui4
    firrtl.matchingconnect %result7, %reg: !firrtl.uint<4>
  }

  // CHECK-LABEL: RegResetImplicitExtOrTrunc
  firrtl.module @RegResetImplicitExtOrTrunc(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, out %out: !firrtl.uint<4>) {
    // CHECK: firrtl.regreset
    %c0_ui3 = firrtl.constant 0 : !firrtl.uint<3>
    %r = firrtl.regreset %clock, %reset, %c0_ui3 : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<3>, !firrtl.uint<2>
    %0 = firrtl.cat %r, %r : (!firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<4>
    firrtl.matchingconnect %r, %r : !firrtl.uint<2>
    firrtl.matchingconnect %out, %0 : !firrtl.uint<4>
  }
}
