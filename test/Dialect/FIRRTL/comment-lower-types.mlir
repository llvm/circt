// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-types))' %s | FileCheck %s

firrtl.circuit "CommentLowerTypes" {
  // CHECK-LABEL: firrtl.module private @Child
  // CHECK-SAME: attributes {comment = "child module comment"}
  firrtl.module private @Child(out %o: !firrtl.bundle<a: uint<1>, b: uint<2>>) attributes {comment = "child module comment"} {}

  // CHECK-LABEL: firrtl.module @CommentLowerTypes
  firrtl.module @CommentLowerTypes(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %i: !firrtl.bundle<a: uint<1>, b: uint<2>>) {
    // CHECK: %wire_a = firrtl.wire
    // CHECK-SAME: comment = "wire comment"
    // CHECK: %wire_b = firrtl.wire
    // CHECK-SAME: comment = "wire comment"
    %wire = firrtl.wire {comment = "wire comment"} : !firrtl.bundle<a: uint<1>, b: uint<2>>

    // CHECK: %node_a = firrtl.node
    // CHECK-SAME: comment = "node comment"
    // CHECK: %node_b = firrtl.node
    // CHECK-SAME: comment = "node comment"
    %node = firrtl.node %i {comment = "node comment"} : !firrtl.bundle<a: uint<1>, b: uint<2>>

    // CHECK: %reg_a = firrtl.reg %clock
    // CHECK-SAME: comment = "reg comment"
    // CHECK: %reg_b = firrtl.reg %clock
    // CHECK-SAME: comment = "reg comment"
    %reg = firrtl.reg %clock {comment = "reg comment"} : !firrtl.clock, !firrtl.bundle<a: uint<1>, b: uint<2>>

    // CHECK: %regreset_a = firrtl.regreset %clock, %reset
    // CHECK-SAME: comment = "regreset comment"
    // CHECK: %regreset_b = firrtl.regreset %clock, %reset
    // CHECK-SAME: comment = "regreset comment"
    %regreset = firrtl.regreset %clock, %reset, %i {comment = "regreset comment"} : !firrtl.clock, !firrtl.uint<1>, !firrtl.bundle<a: uint<1>, b: uint<2>>, !firrtl.bundle<a: uint<1>, b: uint<2>>

    // CHECK: %child_o_a, %child_o_b = firrtl.instance child
    // CHECK-SAME: comment = "instance comment"
    %child_o = firrtl.instance child {comment = "instance comment"} @Child(out o: !firrtl.bundle<a: uint<1>, b: uint<2>>)
  }
}
