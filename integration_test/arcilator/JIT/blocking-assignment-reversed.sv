// RUN: circt-verilog %s | arcilator --run --jit-entry=BlockingAssignmentReversed_main | FileCheck --match-full-lines %s
// REQUIRES: arcilator-jit

// Reverse the processes in blocking-assignment.sv to ensure that scheduling
// follows signal assignments rather than the order of the processes in the IR.
// CHECK: q=2 r=2
// CHECK-NEXT: q=3 r=3
// CHECK-NOT: q=

module BlockingAssignmentReversed;
  logic [7:0] a = 1;
  logic [7:0] q = 0;
  logic [7:0] r = 0;
  always @(q or r)
    $display("q=%0d r=%0d", q, r);
  always @(a) begin
    q = a;
    r = q;
  end
  initial begin
    #5 a = 2;
    #10 a = 3;
  end
endmodule
