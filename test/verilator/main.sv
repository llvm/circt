// REQUIRES: verilator
// RUN: verilator --cc --top-module main -Wall %s

module main(
  input logic clk,
  input wire rst_n,

  output logic [15:0] x
);

  reg [15:0] x_int;
  assign x = x_int << 2;

  always@(posedge clk)
  begin
    if (~rst_n)
    begin
      x_int <= 16'h0;
    end else begin
      x_int <= x_int + 1;
    end
  end

endmodule
