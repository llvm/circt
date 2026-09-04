// REQUIRES: slang
// REQUIRES: libz3
// REQUIRES: circt-bmc-jit
// UNSUPPORTED: valgrind
// RUN: circt-verilog %s --top=Top | circt-bmc - --module Top -b 1 --shared-libs=%libz3 | FileCheck %s

// CHECK: counterexample for Top:
// CHECK: cycle 0:
// CHECK: Assertion can be violated!

module Top(
    input logic clk,
    input logic data,
    output logic sampled
);
  always_ff @(posedge clk)
    sampled <= data;

  assert property (@(posedge clk) data);
endmodule
