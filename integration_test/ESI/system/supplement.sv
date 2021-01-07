// Auxiliary file for tests in this directory. Contains hand-coded modules for
// use in the ESI systems under test.

// Produce a stream of incrementing integers.
module IntCountProd (
  input clk,
  input rstn,
  IValidReady_i32.sink ints
);
  logic unsigned [31:0] count;
  assign ints.valid = rstn;
  assign ints.data = count;

  always@(posedge clk) begin
    if (~rstn)
      count <= 32'h0;
    else if (ints.ready)
      count <= count + 1'h1;
  end
endmodule

// Accumulate a stream of integers. Randomly backpressure. Print status every
// cycle. Output the total over a raw port.
module IntAcc (
  input clk,
  input rstn,
  IValidReady_i32.source ints,
  output unsigned [31:0] totalOut
);
  logic unsigned [31:0] total;

  // De-assert ready randomly
  int unsigned randReady;
  assign ints.ready = rstn && (randReady > 25);

  always@(posedge clk) begin
    randReady <= $urandom_range(100, 0);
    if (~rstn) begin
      total <= 32'h0;
    end else begin
      $display("Total: %10d", total);
      $display("Data: %5d", ints.data);
      if (ints.valid && ints.ready)
        total <= total + ints.data;
    end
  end
endmodule
