// REQUIRES: slang
// REQUIRES: libz3
// REQUIRES: circt-bmc-jit
// UNSUPPORTED: valgrind

// Declaration initializers must reach the generated registers and constrain
// the initial BMC state.
// RUN: circt-verilog %s --top=Zero | FileCheck %s --check-prefix=ZERO-IR
// RUN: circt-verilog %s --top=Nonzero | FileCheck %s --check-prefix=NONZERO-IR
// RUN: circt-verilog %s --top=Uninitialized | FileCheck %s --check-prefix=UNINITIALIZED-IR
// RUN: circt-verilog %s --top=Zero | circt-bmc - --module Zero -b 1 --shared-libs=%libz3 | FileCheck %s --check-prefix=SAFE
// RUN: circt-verilog %s --top=Zero | circt-bmc - --module Zero -b 4 --shared-libs=%libz3 | FileCheck %s --check-prefix=FAIL
// RUN: circt-verilog %s --top=Zero | circt-bmc - --module Zero -b 1 --rising-clocks-only --shared-libs=%libz3 | FileCheck %s --check-prefix=SAFE
// RUN: circt-verilog %s --top=Nonzero | circt-bmc - --module Nonzero -b 1 --shared-libs=%libz3 | FileCheck %s --check-prefix=SAFE
// RUN: circt-verilog %s --top=Nonzero | circt-bmc - --module Nonzero -b 1 --rising-clocks-only --shared-libs=%libz3 | FileCheck %s --check-prefix=SAFE

// A variable without a declaration initializer must remain unconstrained.
// RUN: circt-verilog %s --top=Uninitialized | circt-bmc - --module Uninitialized -b 1 --shared-libs=%libz3 | FileCheck %s --check-prefix=FAIL

// SAFE: Bound reached with no violations!
// FAIL: Assertion can be violated!
// ZERO-IR: seq.firreg {{.*}} preset 0
// NONZERO-IR: seq.firreg {{.*}} preset 7
// UNINITIALIZED-IR: seq.firreg {{.*}} clock {{%[a-zA-Z0-9_]+}} : i4

module Zero(input logic clk);
  logic [3:0] count = 4'd0;
  always_ff @(posedge clk)
    count <= count + 4'd1;
  assert property (@(posedge clk) count < 4'd3);
endmodule

module Nonzero(input logic clk);
  logic [3:0] count = 4'd7;
  always_ff @(posedge clk)
    count <= count + 4'd1;
  assert property (@(posedge clk) count >= 4'd7);
endmodule

module Uninitialized(input logic clk);
  logic [3:0] count;
  always_ff @(posedge clk)
    count <= count + 4'd1;
  assert property (@(posedge clk) count == 4'd0);
endmodule
