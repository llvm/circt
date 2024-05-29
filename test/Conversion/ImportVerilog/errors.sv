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

module Foo;
  // expected-error @below {{unsupported construct}}
  nettype real x;
endmodule

// -----

module Foo;
  // expected-error @+2 {{unsupported type}}
  // expected-note @+1 {{untyped}}
  interconnect x;
endmodule

// -----

// expected-error @below {{unsupported construct}}
package Foo;
endpackage

module Bar;
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

// -----

module Foo;
  // expected-remark @below {{declared here}}
  int a, b;
  initial begin
    // expected-error @below {{replication constant can only be zero inside of a concatenation}}
    a = {0{32'd5}};
    // expected-error @below {{value must be positive}}
    a = {-1{32'd5}};
    // expected-error @below {{value must not have any unknown bits}}
    a = {32'bx{1'b0}};
    // expected-error @below {{value must not have any unknown bits}}
    a = {32'bz{1'b0}};
    // expected-error @below {{reference to non-constant variable 'b' is not allowed in a constant expression}}
    a = {b{32'd5}};
  end
endmodule

// -----

module Foo;
  bit [3:0] a;
  bit [1:0] b;
  // expected-remark @below {{declared here}}
  bit c;
  initial begin
    // expected-error @below {{cannot refer to element 1'bx of 'bit[3:0]' [-Windex-oob]}}
    c = a[1'bx];
    // expected-error @below {{endianness of selection must match declared range (type is 'bit[3:0]')}}
    b = a[0:1];
    // expected-error @below {{reference to non-constant variable 'c' is not allowed in a constant expression}}
    b = a[c:1];
    // expected-error @below {{reference to non-constant variable 'c' is not allowed in a constant expression}}
    b = a[2-:c];
    // expected-error @below {{value must not have any unknown bits}}
    b = a[1'bz:0];
    // expected-error @below {{value must not have any unknown bits}}
    b = a[1-:'x];
    // expected-error @below {{value must be positive}}
    b = a[1-:0];
    // expected-error @below {{value must be positive}}
    b = a[1-:-1];
  end
endmodule

// -----

module Foo;
  int x;
  initial begin
    // expected-remark @below {{declared here}}
    automatic int a;
    // expected-error @below {{nonblocking assignment to automatic variable 'a' is not allowed}}
    a <= x;
    // expected-error @below {{declaration must come before all statements in the block}}
    automatic int b;
  end
endmodule

// -----

module Foo;
  int a, b, c;
  bit d [3:0];
  initial begin
    // expected-error @below {{literals with X or Z bits not supported}}
    c = 'x inside { a };
    // expected-error @below {{literals with X or Z bits not supported}}
    c = a inside { 'z, a };
    // expected-error @below {{literals with X or Z bits not supported}}
    c = a inside { a, ['x:b] };
    // expected-error @below {{literals with X or Z bits not supported}}
    c = a inside { a, [b:'z] };
    // expected-error @below {{unpacked arrays in 'inside' expressions not supported}}
    c = a inside { d };
  end
endmodule
