module AsyncReset(input logic clock, input logic reset, input int d, output int q);
  always_ff @(posedge clock or posedge reset) if (reset) q <= 42; else q <= d;
endmodule
