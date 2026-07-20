// RUN: circt-opt --verify-roundtrip %s | FileCheck %s

firrtl.circuit "Top" {
  // CHECK: firrtl.module @Child() attributes {comment = "module comment"} {
  firrtl.module @Child() attributes {comment = "module comment"} {}

  firrtl.module @Top(
    in %clock: !firrtl.clock,
    in %reset: !firrtl.uint<1>
  ) {
    // CHECK: %wire = firrtl.wire {comment = "wire comment"} : !firrtl.uint<1>
    %wire = firrtl.wire {comment = "wire comment"} : !firrtl.uint<1>

    // CHECK: %node = firrtl.node %wire {comment = "node comment"} : !firrtl.uint<1>
    %node = firrtl.node %wire {comment = "node comment"} : !firrtl.uint<1>

    // CHECK: %reg = firrtl.reg %clock {comment = "reg comment"} : !firrtl.clock, !firrtl.uint<1>
    %reg = firrtl.reg %clock {comment = "reg comment"} : !firrtl.clock, !firrtl.uint<1>

    // CHECK: %regreset = firrtl.regreset %clock, %reset, %wire {comment = "regreset comment"} : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    %regreset = firrtl.regreset %clock, %reset, %wire {comment = "regreset comment"} : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>

    // CHECK: firrtl.instance child {comment = "instance comment"} @Child()
    firrtl.instance child {comment = "instance comment"} @Child()
  }
}
