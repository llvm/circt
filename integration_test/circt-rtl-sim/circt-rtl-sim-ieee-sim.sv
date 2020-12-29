// REQUIRES: ieee-sim
// RUN: circt-rtl-sim.py --sim %ieee-sim --cycles 2 %s

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
