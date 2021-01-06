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

module IntAcc (
  input clk,
  input rstn,
  IValidReady_i32.source ints
);
  logic unsigned [31:0] total;
  assign ints.ready = rstn;

  always@(posedge clk) begin
    if (~rstn) begin
      total <= 32'h0;
    end else begin
      $display("Total: %10d", total);
      $display("Data: %5d", ints.data);
      if (ints.valid)
        total <= total + ints.data;
    end
  end
endmodule
