// RUN: circt-verilog %s | arcilator --run --jit-entry=FixedPointObservers_main > %t
// RUN: FileCheck --match-full-lines --check-prefix=B --implicit-check-not="b=" %s < %t
// RUN: FileCheck --match-full-lines --check-prefix=C --implicit-check-not="c=" %s < %t
// REQUIRES: arcilator-jit

// Neither observer feeds back into the cycle. Both must see every change,
// including through the additional blocking assignment to c. Declare the
// observers before the producers to exercise dependency-driven lowering.
// Check each observer independently: require every event exactly once, in
// order, without imposing an order between the two observers.
// B: b=1
// B: b=2
// B: b=3
// B: b=4
// B: b=5
// B: settled a=5 b=5 c=5
// C: c=1
// C: c=2
// C: c=3
// C: c=4
// C: c=5
// C: settled a=5 b=5 c=5

module FixedPointObservers;
  logic [3:0] limit = 0;
  logic [3:0] a = 0;
  logic [3:0] b = 0;
  logic [3:0] c = 0;

  always @(c) $display("c=%0d", c);
  always @(b) $display("b=%0d", b);
  always @(b) c = b;
  always @(b) a = b;
  always @(a or limit) begin
    if (a < limit) b = a + 1;
    else b = a;
  end

  initial begin
    #5 limit = 3;
    #6 limit = 5;
    #1 $display("settled a=%0d b=%0d c=%0d", a, b, c);
  end
endmodule
