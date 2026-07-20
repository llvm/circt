// RUN: firtool %s --format=mlir --verilog | FileCheck %s

firrtl.circuit "Top" {
  // CHECK:      // child module comment
  // CHECK-NEXT: module Child
  firrtl.module private @Child(
    in %clock: !firrtl.clock,
    in %reset: !firrtl.uint<1>,
    in %in: !firrtl.uint<1>,
    out %out: !firrtl.uint<1>
  ) attributes {comment = "child module comment"} {
    // CHECK:      // wire comment
    // CHECK-NEXT: wire{{.*}}wire
    %wire = firrtl.wire {comment = "wire comment"} : !firrtl.uint<1>
    firrtl.connect %wire, %in : !firrtl.uint<1>, !firrtl.uint<1>

    // CHECK:      // node comment
    // CHECK-NEXT: wire{{.*}}node
    %node = firrtl.node %wire {comment = "node comment"} : !firrtl.uint<1>

    // CHECK:      // reg comment
    // CHECK-NEXT: reg{{.*}}reg
    %reg = firrtl.reg %clock {comment = "reg comment"} : !firrtl.clock, !firrtl.uint<1>
    firrtl.connect %reg, %node : !firrtl.uint<1>, !firrtl.uint<1>

    // CHECK:      // regreset comment
    // CHECK-NEXT: reg{{.*}}regreset
    %regreset = firrtl.regreset %clock, %reset, %in {comment = "regreset comment"} : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %regreset, %reg : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %out, %regreset : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // CHECK:      // top module comment
  // CHECK-NEXT: module Top
  firrtl.module @Top(
    in %clock: !firrtl.clock,
    in %reset: !firrtl.uint<1>,
    in %in: !firrtl.uint<1>,
    out %out: !firrtl.uint<1>
  ) attributes {comment = "top module comment"} {
    // CHECK:      // instance comment
    // CHECK-NEXT: Child child
    %child_clock, %child_reset, %child_in, %child_out = firrtl.instance child {comment = "instance comment"} @Child(in clock: !firrtl.clock, in reset: !firrtl.uint<1>, in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    firrtl.connect %child_clock, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %child_reset, %reset : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %child_in, %in : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %out, %child_out : !firrtl.uint<1>, !firrtl.uint<1>
  }
}
