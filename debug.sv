module AsyncReset(
  input logic clock,
  input logic reset,
  input logic enable,
  input int d,
  output int q1,
  output int q2,
  output int q3,
  output int q4
);
  always_ff @(posedge clock or posedge reset) if (reset) q1 <= 42; else if (enable) q1 <= d;
  always_ff @(posedge clock or posedge reset) if (reset) q2 <= 42; else q2 <= d;
  always_ff @(posedge clock or posedge reset) q3 <= d;
  always_ff @(posedge clock or posedge reset) q4 <= reset ? 42 : d;
endmodule
