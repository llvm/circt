// RUN: circt-translate --import-verilog --verify-diagnostics --split-input-file %s
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

// expected-error @below {{expected ';'}}
module Foo 4;
endmodule

// -----

// expected-note @below {{expanded from macro 'FOO'}}
`define FOO input
// expected-note @below {{expanded from macro 'BAR'}}
`define BAR `FOO
// expected-error @below {{expected identifier}}
module Bar(`BAR);
endmodule

// -----

module Foo;
  mailbox a;
  string b;
  // expected-error @below {{value of type 'string' cannot be assigned to type 'mailbox'}}
  initial a = b;
endmodule

// -----

module Foo;
  // expected-error @below {{unsupported construct}}
  genvar a;
endmodule

// -----

module Foo;
  int x;
  // expected-error @below {{delayed assignments not supported}}
  initial x <= #1ns x;
endmodule

// -----

module Foo;
  int x;
  // expected-error @below {{delayed continuous assignments not supported}}
  assign #1ns x = x;
endmodule

// -----

module Foo;
  int a;
  initial begin
    // expected-error @below {{unsupported statement}}
    release a;
  end
endmodule

// -----

module Foo;
  bit x, y;
  // expected-error @below {{match patterns in if conditions not supported}}
  initial if (x matches 42) x = y;
endmodule

// -----

module Foo;
  bit y;
  // expected-error @below {{variables in for loop initializer not supported}}
  initial for (bit x = 0; x;) x = y;
endmodule

// -----

module Foo;
  logic x;
  // expected-error @below {{literals with X or Z bits not supported}}
  initial x = 'x;
  // expected-error @below {{literals with X or Z bits not supported}}
  initial x = 'z;
endmodule
