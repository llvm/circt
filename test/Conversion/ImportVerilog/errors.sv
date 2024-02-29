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

// -----

module Foo;
  initial begin
    // expected-error @below {{variable initializer expressions not supported}}
    automatic int a = 0;
  end
endmodule

// -----

module Foo;
  int a;
  initial begin
    // expected-error @below {{unsupported statement}}
    release a;
  end
endmodule
