// REQUIRES: verilator
// RUN: circt-rtl-sim.py --cycles 2 %s | FileCheck %s
// This also works if you have Questa available:
//   circt-rtl-sim.py --sim %comsim --cycles 2 %s

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
