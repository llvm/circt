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

  // CHECK-LABEL: @CommentsBlockRemoval
  firrtl.module @CommentsBlockRemoval(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, out %reg_out: !firrtl.uint<1>, out %regreset_out: !firrtl.uint<1>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>

    // CHECK: %commented_reg = firrtl.reg %clock
    // CHECK-SAME: comment = "register optimizer reg comment"
    %commented_reg = firrtl.reg %clock {comment = "register optimizer reg comment"} : !firrtl.clock, !firrtl.uint<1>
    // CHECK: firrtl.matchingconnect %commented_reg, %c0_ui1
    firrtl.matchingconnect %commented_reg, %c0_ui1 : !firrtl.uint<1>
    firrtl.matchingconnect %reg_out, %commented_reg : !firrtl.uint<1>

    // CHECK: %commented_regreset = firrtl.regreset %clock, %reset, %c0_ui1
    // CHECK-SAME: comment = "register optimizer regreset comment"
    %commented_regreset = firrtl.regreset %clock, %reset, %c0_ui1 {comment = "register optimizer regreset comment"} : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: firrtl.matchingconnect %commented_regreset, %c0_ui1
    firrtl.matchingconnect %commented_regreset, %c0_ui1 : !firrtl.uint<1>
    firrtl.matchingconnect %regreset_out, %commented_regreset : !firrtl.uint<1>
  }

  // Comments do not make an otherwise dead feedback loop live.
  // CHECK-LABEL: @DeadComments
  // CHECK-NOT: %dead_reg =
  // CHECK-NOT: %dead_regreset =
  // CHECK-NOT: dead optimizer reg comment
  // CHECK-NOT: dead optimizer regreset comment
  firrtl.module private @DeadComments(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>

    %dead_reg = firrtl.reg %clock {comment = "dead optimizer reg comment"} : !firrtl.clock, !firrtl.uint<1>
    firrtl.matchingconnect %dead_reg, %dead_reg : !firrtl.uint<1>

    %dead_regreset = firrtl.regreset %clock, %reset, %c0_ui1 {comment = "dead optimizer regreset comment"} : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.matchingconnect %dead_regreset, %dead_regreset : !firrtl.uint<1>
  }
}
