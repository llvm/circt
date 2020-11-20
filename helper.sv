`include "xil_primitives.sv"
`ifndef HIR_HELPER
`define HIR_HELPER

module weighted_sum (
  output wire[31:0] out,
  input wire[31:0] in1,
  input wire[31:0] w1,
  input wire[31:0] in2,
  input wire[31:0] w2,
  input wire tstart,
  input wire clk
);
reg[31:0] m1_reg;
reg[31:0] m2_reg;
always@(posedge clk) begin
  m1_reg <= in1*w1;
  m2_reg <= in2*w2;
end
assign out = m1_reg+m2_reg;
endmodule


module max (
  output reg[31:0] out,
  input wire[31:0] in1,
  input wire[31:0] in2,
  input wire tstart,
  input wire clk
);
always@(posedge clk) begin
  out <= (in1>in2)?in1:in2;
end
endmodule

module add(
  output reg[31:0] out,
  input  wire[31:0] in1,
  input  wire[31:0] in2,
  input wire tstart,
  input wire clk
);
always@(posedge clk) begin
  out <= in1+in2;
end
endmodule

module mult(
  output reg[31:0] out,
  input  wire[31:0] in1,
  input  wire[31:0] in2,
  input wire tstart,
  input wire clk
);
reg[31:0] out1;
always@(posedge clk) begin
  out1 <= in1*in2;
  out <= out1;
end
endmodule
`endif
