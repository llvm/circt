// RUN: circt-verilog %s | arcilator --run --jit-entry=BlockingAssignmentChain_main | FileCheck --match-full-lines %s
// REQUIRES: arcilator-jit

// Each assignment must trigger the next process in the same evaluation, even
// though the processes appear in reverse dependency order.
// CHECK: q=2 r=3
// CHECK-NEXT: q=3 r=4
// CHECK-NOT: q=

module BlockingAssignmentChain;
  logic [7:0] a = 1;
  logic [7:0] q = 0;
  logic [7:0] r = 0;
  always @(r)
    $display("q=%0d r=%0d", q, r);
  always @(q)
    r = q + 1;
  always @(a)
    q = a;
  initial begin
    #5 a = 2;
    #10 a = 3;
  end
endmodule
