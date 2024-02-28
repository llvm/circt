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

module Foo(
  // expected-error @below {{unsupported module port}}
  input a
);
endmodule

// -----

// expected-error @below {{unsupported construct}}
package Foo;
endpackage

module Bar;
endmodule

// -----

module Foo;
  // expected-error @below {{variable initializer expressions not supported}}
  int a = 0;
endmodule
