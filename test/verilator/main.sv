// REQUIRES: verilator
// RUN: verilator --cc --top-module main -Wall -Wpedantic %s

// Tell Verilator not to complain about multiple modules in same file.
/* verilator lint_off DECLFILENAME */

module main(
  input logic clk,
  input wire rst_n,

  output logic [15:0] x
);

  reg [15:0] x_int;
  assign x = (x_int << 2) ^ {fooOut, barOut};

  always@(posedge clk) begin
    if (~rst_n) begin
      x_int <= 16'h0;
    end else begin
      x_int <= x_int + 1;
    end
  end

  logic [3:0] fooIn = x[5:2];
  logic [7:0] fooOut;
  ParameterizedModule #(.INWIDTH(4), .OUTWIDTH(8)) foo (.a(fooIn), .x(fooOut));

  logic [9:0] barIn = x[11:2];
  logic [7:0] barOut;
  ParameterizedModule #(.INWIDTH(10), .OUTWIDTH(8)) bar (.a(barIn), .x(barOut));
endmodule

module ParameterizedModule # (
  parameter INWIDTH = 4,
  parameter OUTWIDTH = 4
) (
  input logic [INWIDTH-1:0] a,
  output logic [OUTWIDTH-1:0] x
);

  generate
    if (OUTWIDTH > INWIDTH) begin
      localparam DIFF = OUTWIDTH - INWIDTH;
      assign x = {a, {DIFF{1'b0}}};
    end else begin
      if (OUTWIDTH != INWIDTH) begin
        reg [63:0] leftOver;

        always@(a) begin
          leftOver = leftOver + {{(63 - INWIDTH + OUTWIDTH){1'b0}},
                                 a[INWIDTH - 1: OUTWIDTH]};
        end
      end

      assign x = a[OUTWIDTH-1:0];
    end
  endgenerate

endmodule
