// REQUIRES: slang
// REQUIRES: libz3
// REQUIRES: circt-bmc-jit
// UNSUPPORTED: valgrind
// RUN: circt-bmc %s --module Top -b 4 --shared-libs=%libz3 --print-only-first-counterexample | FileCheck %s
// RUN: circt-bmc - --module Top -b 4 --shared-libs=%libz3 --print-only-first-counterexample < %s | FileCheck %s

// CHECK: counterexample for Top:
// CHECK: cycle 0:
// CHECK:   state_next = 0x1
// CHECK:   state_state = 0x0
// CHECK: cycle 1:
// CHECK:   state_next = 0x
// CHECK:   state_state = 0x1
// CHECK: Assertion can be violated!

module Top(input logic clk, input logic data);
  logic state = 1'b0;

  always_ff @(posedge clk)
    state <= data;

  assert property (@(posedge clk) state == 1'b0);
endmodule
