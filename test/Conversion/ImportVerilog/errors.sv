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
  bit y;
  // expected-error @below {{unsupported non-blocking assignment timing control: SignalEvent}}
  initial x <= @y x;
endmodule

// -----
module Foo;
  int x;
  // expected-error @below {{implicit events cannot be used here}}
  initial x = @* x;
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
  int a, b[3];
  // expected-error @below {{unpacked arrays in 'inside' expressions not supported}}
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
function Foo;
  // expected-error @below {{unsupported format specifier `%l`}}
  $write("%l");
endfunction

// -----
function Foo;
  // expected-error @below {{string format specifier with width not supported}}
  $write("%42s", "foo");
endfunction

// -----
function time Foo;
  // expected-error @below {{time value is larger than 18446744073709549568 fs}}
  return 100000s;
endfunction

// -----
module Foo;
  // expected-error @below {{unsupported type: associative arrays with wildcard index}}
  int x[*];
endmodule

// -----
function void foo;
  struct packed { time t; } a;
  int b;
  // expected-error @below {{contains a time type}}
  a = b;
endfunction

// -----
function void foo;
  int a;
  struct packed { time t; } b;
  // expected-error @below {{contains a time type}}
  a = b;
endfunction

// -----
module Foo;
  logic a;
  string b;
  // expected-error @below {{expected integer argument for `$past`}}
  assert property (@(posedge a) $past(b));
endmodule

// -----
module Foo;
  int a;
  // expected-error @below {{sequence has no explicit clocking event and one cannot be inferred from context}}
  assert property (a);
endmodule

// -----
module Foo;
  typedef enum { A, B, C } e;
  e val;
  initial begin
    // expected-error @below {{unsupported system call `next`}}
    val = val.next();
  end
endmodule

// -----
module Foo;
  typedef enum { A, B, C } e;
  e val;
  initial begin
    // expected-error @below {{unsupported system call `prev`}}
    val = val.prev();
  end
endmodule

// -----
module Foo;
  int inp[];
  int tmp[];
  initial begin
    // expected-error @below {{unsupported system call `$size`}}
    tmp = new[$size(inp)];
  end
endmodule

// -----
// Cross-module hierarchical references can produce null port values that must
// not crash during instance creation. This is an error in the hierarchical
// reference resolution code that actually needs fixing. This test guards
// against a regression to this being a crash.
module HierRefTop(input i, output o);
  // expected-error @below {{unsupported port}}
  HierRefA A();
  HierRefB B();
  assign A.i = i;
  assign o = B.o;
endmodule
module HierRefA;
  wire i, y;
  assign B.x = !i;
  assign y = !B.y;
endmodule
module HierRefB;
  wire x, y, o;
  assign y = x, o = A.y;
endmodule

// -----
module Foo;
  reg i;
  wire o;
  // expected-error @below {{unsupported delay with rise/fall/turn-off}}
  assign #(1, 2) o = i;
endmodule

// -----
function Foo;
  logic [1:0] a;
  // expected-error @below {{unsupported system call `$fwrite`}}
  $fwrite(32'h0, "%x", a);
endfunction

// -----
module Foo;
  string s;
  byte b;
  initial begin
    // expected-error @below {{string index assignment not supported}}
    s[0] = b;
  end
endmodule

// -----
interface Inner;
  logic x;
endinterface

interface Outer;
  Inner nested();
  logic y;
endinterface

module UsesOuter;
  // expected-error @below {{nested interface instances are not supported: `nested` inside `o`}}
  Outer o();
endmodule

// -----
module Foo;
	int v = 1;

  // expected-error @+2 {{cannot mix continuous and procedural assignments to variable 'v'}}
  // expected-remark @-3 {{also assigned here}}
	assign v = 12;
endmodule

// -----
module Foo;
	int v;

  // expected-error @+3 {{cannot have multiple continuous assignments to variable 'v'}}
  // expected-remark @below {{also assigned here}}
	assign v = 12;
	assign v = 13;
endmodule

// -----
module Foo;
	wire clk = 0;
	int v;

  // expected-error @+3 {{cannot mix continuous and procedural assignments to variable 'v'}}
  // expected-remark @below {{also assigned here}}
	assign v = 12;
	always @(posedge clk) v <= ~v;
endmodule

// -----
module Foo;
	wire clk = 0;
	int v;

  // expected-error @+3 {{cannot mix continuous and procedural assignments to variable 'v'}}
  // expected-remark @below {{also assigned here}}
	always @(posedge clk) v <= ~v;
	assign v = 12;
endmodule

// -----
module Foo;
  logic a;

  // expected-error @below {{'always' procedure does not advance time and so will create a simulation deadlock}}
  always a = ~a;
endmodule
