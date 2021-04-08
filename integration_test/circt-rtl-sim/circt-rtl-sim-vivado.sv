// REQUIRES: vivado
// RUN: circt-rtl-sim.py --sim %xsim% --cycles 2 %s

module top(
  input clk,
  input rstn
);

  always@(posedge clk)
    if (rstn)
      $display("tock");
  // CHECK:      tock
  // CHECK-NEXT: tock

endmodule
