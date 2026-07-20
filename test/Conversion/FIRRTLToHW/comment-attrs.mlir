// RUN: circt-opt -pass-pipeline='builtin.module(lower-firrtl-to-hw)' %s | FileCheck %s

firrtl.circuit "CommentAttrs" {
  // CHECK-LABEL: hw.module private @Child()
  // CHECK-SAME: attributes {comment = "module comment"}
  firrtl.module private @Child() attributes {comment = "module comment"} {}

  // CHECK-LABEL: hw.module @CommentAttrs
  firrtl.module @CommentAttrs(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %i: !firrtl.uint<1>, out %o: !firrtl.uint<1>) {
    // CHECK: %wire = hw.wire
    // CHECK-SAME: comment = "wire comment"
    %wire = firrtl.wire {comment = "wire comment"} : !firrtl.uint<1>
    firrtl.matchingconnect %wire, %i : !firrtl.uint<1>

    // A node with a declaration comment must materialize as its own hw.wire.
    // CHECK: %node = hw.wire %wire
    // CHECK-SAME: comment = "node comment"
    %node = firrtl.node %wire {comment = "node comment"} : !firrtl.uint<1>

    // CHECK: %reg = seq.firreg
    // CHECK-SAME: comment = "reg comment"
    %reg = firrtl.reg %clock {comment = "reg comment"} : !firrtl.clock, !firrtl.uint<1>
    firrtl.matchingconnect %reg, %node : !firrtl.uint<1>

    // CHECK: %regreset = seq.firreg
    // CHECK-SAME: comment = "regreset comment"
    %regreset = firrtl.regreset %clock, %reset, %i {comment = "regreset comment"} : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.matchingconnect %regreset, %reg : !firrtl.uint<1>
    firrtl.matchingconnect %o, %regreset : !firrtl.uint<1>

    // CHECK: hw.instance "child" @Child()
    // CHECK-SAME: comment = "instance comment"
    firrtl.instance child {comment = "instance comment"} @Child()

    // Zero-width declarations remain terminally removable even with comments.
    // CHECK-NOT: zero_width
    // CHECK-NOT: zero width comment
    %zero_width = firrtl.wire {comment = "zero width comment"} : !firrtl.uint<0>
  }
}
