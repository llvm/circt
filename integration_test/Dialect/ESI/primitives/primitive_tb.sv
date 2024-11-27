// REQUIRES: ieee-sim
// RUN: circt-rtl-sim.py --sim %ieee-sim %esi_prims %s

//===- primitive_tb.sv - tests for ESI primitives -----------*- verilog -*-===//
//
// Testbenches for ESI primitives. Since these rely on an IEEE SystemVerilog
// simulator (Verilator is not a simulator by the IEEE definition) we don't have
// a way to run them as part of our PR gate. They're here for posterity.
//
//===----------------------------------------------------------------------===//

`define assert_fatal(pred) \
  if (!(pred)) \
    $fatal();

module top (
  input logic clk,
  input logic rst
);

  logic a_valid = 0;
  logic [7:0] a = 0;
  logic a_ready;

  logic x_valid;
  logic [7:0] x;
  logic x_ready = 0;


  ESI_PipelineStage s1 (
    .clk(clk),
    .rst(rst),
    .a_valid(a_valid),
    .a(a),
    .a_ready(a_ready),
    .x_valid(x_valid),
    .x(x),
    .x_ready(x_ready)
  );

  // Increment the input every cycle.
  always begin
    @(posedge clk) #1;
    if (~rst)
      a++;
  end

  // Track the number of tokens currently in the stage for debugging.
  int balance = 0;
  always@(posedge clk) begin
    if (x_valid && x_ready)
      balance--;
    if (a_valid && a_ready)
      balance++;
  end

  initial begin
    // Wait until rst is deasserted.
    @(negedge rst);
    @(posedge clk);

    a_valid = 1;
    `assert_fatal (a_ready);
    @(posedge clk) #1;
    `assert_fatal (x_valid);
    `assert_fatal (x == 8'h02);
    `assert_fatal (a_ready);

    a_valid = 1;
    @(posedge clk) #1;
    `assert_fatal (x_valid);
    `assert_fatal (x == 8'h02);
    `assert_fatal (~a_ready);
    a_valid = 1;

    @(posedge clk) #1;
    `assert_fatal (x_valid);
    `assert_fatal (x == 8'h02);
    `assert_fatal (~a_ready);
    x_ready = 1;

    @(posedge clk) #1;
    `assert_fatal (x_valid);
    `assert_fatal (x == 8'h03);
    `assert_fatal (a_ready);
    x_ready = 1;

    @(posedge clk) #1;
    `assert_fatal (x_valid);
    `assert_fatal (x == 8'h06);
    `assert_fatal (a_ready);
    x_ready = 0;

    @(posedge clk) #1;
    `assert_fatal (x_valid);
    `assert_fatal (x == 8'h06);
    `assert_fatal (~a_ready);
    x_ready = 1;

    @(posedge clk) #1;
    `assert_fatal (x_valid);
    `assert_fatal (x == 8'h07);
    `assert_fatal (a_ready);
    x_ready = 1;
    a_valid = 0;

    @(posedge clk) #1;
    `assert_fatal (~x_valid);
    `assert_fatal (a_ready);
    x_ready = 1;
    a_valid = 0;

    @(posedge clk) #1;
    `assert_fatal (~x_valid);
    `assert_fatal (a_ready);
    x_ready = 1;
    a_valid = 0;

    @(posedge clk) #1;
    `assert_fatal (~x_valid);
    `assert_fatal (a_ready);
    x_ready = 1;
    a_valid = 0;

    @(posedge clk) #1;
    `assert_fatal (~x_valid);
    `assert_fatal (a_ready);
    x_ready = 1;
    a_valid = 1;

    @(posedge clk) #1;
    `assert_fatal (x_valid);
    `assert_fatal (x == 8'h0D);
    `assert_fatal (a_ready);
    x_ready = 1;
    a_valid = 1;

    @(posedge clk) #1;
    `assert_fatal (x_valid);
    `assert_fatal (x == 8'h0E);
    `assert_fatal (a_ready);
    x_ready = 1;
    a_valid = 1;

    @(posedge clk) #1;
    `assert_fatal (x_valid);
    `assert_fatal (x == 8'h0F);
    `assert_fatal (a_ready);
    x_ready = 1;
    a_valid = 1;

    @(posedge clk) #1;
    @(posedge clk) #1;
    @(posedge clk) #1;
    @(posedge clk) #1;
    @(posedge clk) #1;
    @(posedge clk) #1;
    $finish();
  end

endmodule
