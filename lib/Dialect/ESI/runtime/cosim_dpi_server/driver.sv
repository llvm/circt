//===- driver.sv - ESI cosim testbench driver -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the top module driver for an ESI cosimulation. It simply provides a
// clock and reset signal. In the future, the reset signal will become a reset
// handshake.
//
//===----------------------------------------------------------------------===//

`timescale 1ns / 100ps

`ifndef TOP_MODULE
`define TOP_MODULE ESI_Cosim_Top
`endif

module driver();

  logic clk = 0;
  logic rst = 0;

  `TOP_MODULE top (
    .clk(clk),
    .rst(rst)
  );

  always begin
    // A clock period is #4.
    clk = ~clk;
    #2;
  end

  initial begin
    int cycles;

    $display("[driver] Starting simulation");

    rst = 1;
    // Hold in reset for 4 cycles.
    @(posedge clk);
    @(posedge clk);
    @(posedge clk);
    @(posedge clk);
    rst = 0;

    if ($value$plusargs ("cycles=%d", cycles)) begin
      int i;
      for (i = 0; i < cycles; i++) begin
        @(posedge clk);
      end
      $display("[driver] Ending simulation at tick #%0d", $time);
      $finish();
    end
  end

endmodule
