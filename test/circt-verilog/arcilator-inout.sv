// RUN: circt-verilog --ir-hw %s | arcilator --disable-output

module MixedPorts(
  inout wire c,
  input logic clk,
  output logic q
);
  always_ff @(posedge clk) begin
    q <= c;
  end
endmodule
