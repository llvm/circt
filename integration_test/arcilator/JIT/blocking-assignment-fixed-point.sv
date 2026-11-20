// RUN: circt-verilog %s | arcilator --run --jit-entry=BlockingAssignmentFixedPoint_main | FileCheck --match-full-lines %s
// REQUIRES: arcilator-jit

// The update at time 5 requires several rounds through the two mutually
// triggering processes before both signals converge to 3.
// CHECK: a=3 b=3
// CHECK-NOT: a=

module BlockingAssignmentFixedPoint;
  logic [3:0] limit = 0;
  logic [3:0] a = 0;
  logic [3:0] b = 0;

  always @(a or limit) begin
    if (a < limit)
      b = a + 1;
    else
      b = a;
  end

  always @(b)
    a = b;

  initial begin
    #5 limit = 3;
    #1 $display("a=%0d b=%0d", a, b);
  end
endmodule
