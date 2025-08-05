// module Foo;
//   timeunit 1ns / 10ps;
//   time a0 = 12ns;
//   time a1 = 2.345ns;
//   realtime b0 = 34ns;
//   realtime b1 = 4.567ns;
// endmodule

// module Bar;
//   timeunit 100ps / 10ps;
//   time a0 = 12ns;
//   time a1 = 2.345ns;
//   realtime b0 = 34ns;
//   realtime b1 = 4.567ns;
// endmodule

module TimeA;
  timeunit 1ns / 1ps;
  time a = 234ns;
  realtime b = 5.67ns;
  time x;
  realtime y;
  TimeB child (.a, .b, .x, .y);
endmodule

module TimeB (input time a, input realtime b, output time x, output realtime y);
  timeunit 100ps / 1ps;
  assign x = 123ns;
  assign y = 4.56ns;
  bit [63:0] u = a;
endmodule
