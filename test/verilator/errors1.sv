// REQUIRES: verilator
// RUN: not verilator --cc --top-module main -Wall -Wpedantic %s 2>&1 | FileCheck %s

// Tell Verilator not to complain about multiple modules in same file.
/* verilator lint_off DECLFILENAME */

module main(
  input logic clk,
  input wire rst_n,
// CHECK: %Warning-UNUSED: {{.*}}:9{{.*}} Signal is not used: 'rst_n'
  output logic [15:0] x
);

  reg [15:0] x_int;
  assign x = (x_int << 2);

  always@(posedge clk) begin
    x_int = x_int + 1;
// CHECK: %Warning-BLKSEQ: {{.*}}:18{{.*}} Blocking assignments (=) in sequential (flop or latch) block
  end
endmodule
