// RUN: circt-verilog %s | circt-opt --lower-seq-to-sv --export-verilog | FileCheck %s
// REQUIRES: slang
// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

// Ensure we can round-trip the following two different flavors of enables on a
// register and have them come out unchanged. These have subtly different
// semantics when `en` is X, which must be preserved in order to satisfy logical
// equivalence checks.

// CHECK-LABEL: module Foo
module Foo (input logic clk, en, d, output logic qa, qb);
  // CHECK: always @(posedge clk)
  always @(posedge clk) begin
    // CHECK: if (en)
    // CHECK:   [[QA:qa.*]] <= d;
    if (en)
      qa <= d;

    // CHECK: [[QB:qb.*]] <= en ? d : [[QB]];
    qb <= en ? d : qb;
  end
endmodule
