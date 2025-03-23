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
  // expected-error @below {{unsupported module member}}
  nettype real x;
endmodule

// -----
module Foo;
  // expected-error @+2 {{unsupported type}}
  // expected-note @+1 {{untyped}}
  interconnect x;
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
  // expected-error @below {{unsupported statement}}
  initial release a;
endmodule

// -----
module Foo;
  bit x, y;
  // expected-error @below {{match patterns in if conditions not supported}}
  initial if (x matches 42) x = y;
endmodule

// -----
module Foo;
  int a, b[];
  // expected-error @below {{unsized unpacked arrays in 'inside' expressions not supported}}
  int c = a inside { b };
endmodule

// -----
module Foo;
  int a, b, c;
  int j;
  initial begin
    // expected-error @below {{streaming operator target size 32 does not fit source size 96}}
    j = {>>{ a, b, c }}; // error: j is 32 bits < 96 bits
  end
endmodule


// -----
module Foo;
  int a, b, c;
  int j;
  initial begin
    // expected-error @below {{streaming operator target size 96 does not fit source size 23}}
    {>>{ a, b, c }} = 23'b1;
  end
endmodule

// -----
module Foo;
  initial begin
    logic [15:0] vec_0;
    logic [47:0] vec_1;
    logic arr [63:0];
    int c;
    // expected-error @below {{Moore only support streaming concatenation with fixed size 'with expression'}}
    vec_1 = {<<byte{vec_0, arr with [c:0]}};
  end
endmodule

// -----
module Foo;
  initial begin
    int my_queue[];
    logic [31:0] vec_0;
    // expected-error @below {{expression of type '!moore.open_uarray<i32>' cannot be cast to a simple bit vector}}
    vec_0 = {<<byte{my_queue}};
  end
endmodule

// -----
module Foo;
  // expected-remark @below {{hello}}
  $info("hello");
  // expected-warning @below {{hello}}
  $warning("hello");
endmodule

// -----
module Foo;
  // expected-error @below {{hello}}
  $error("hello");
endmodule

// -----
module Foo;
  // expected-error @below {{hello}}
  $fatal(0, "hello");
endmodule

// -----
module Top; endmodule
function Foo;
  // expected-error @below {{unsupported format specifier `%l`}}
  $write("%l");
endfunction

// -----
module Top; endmodule
function Foo;
  // expected-error @below {{string format specifier with width not supported}}
  $write("%42s", "foo");
endfunction
