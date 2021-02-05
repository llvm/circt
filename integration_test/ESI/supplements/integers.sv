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

// Accumulate a stream of integers. Print status every cycle. Output the total
// over an ESI channel.
module IntAccNoBP (
  input clk,
  input rstn,
  IValidReady_i32.source ints,
  IValidReady_i32.sink totalOut
);
  logic unsigned [31:0] total;
  assign totalOut.data = total;

  always@(posedge clk) begin
    if (~rstn) begin
      total <= 32'h0;
      ints.ready <= 1;
      totalOut.valid <= 0;
    end else begin
      if (ints.valid && ints.ready) begin
        total <= total + ints.data;
        totalOut.valid <= 1;
        ints.ready <= totalOut.ready;
        $display("Total: %10d", total);
        $display("Data: %5d", ints.data);
      end else if (totalOut.valid && totalOut.ready) begin
        totalOut.valid <= 0;
        ints.ready <= 1;
      end
    end
  end
endmodule

module IntArrSum (
  input clk,
  input rstn,
  IValidReady_ArrayOf2xsi13.source arr,
  IValidReady_i32.sink totalOut
);

  assign totalOut.valid = arr.valid;
  assign arr.ready = totalOut.ready;
  assign totalOut.data = 32'($signed(arr.data[0])) + 32'($signed(arr.data[1]));
endmodule
