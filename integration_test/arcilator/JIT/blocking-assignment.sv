// RUN: circt-verilog %s | arcilator --run --jit-entry=BlockingAssignment_main | FileCheck --match-full-lines %s
// REQUIRES: arcilator-jit

// CHECK: q=2 r=2
// CHECK-NEXT: q=3 r=3
// CHECK-NOT: q=

module BlockingAssignment;
  logic [7:0] a = 1;
  logic [7:0] q = 0;
  logic [7:0] r = 0;
  initial begin
    #5 a = 2;
    #10 a = 3;
  end
  always @(a) begin
    q = a;
    r = q;
  end
  always @(q or r)
    $display("q=%0d r=%0d", q, r);
endmodule

