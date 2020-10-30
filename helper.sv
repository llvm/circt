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

