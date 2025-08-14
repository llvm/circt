// RUN: circt-verilog %s %s
// REQUIRES: slang

// Listing a file twice caused the Slang's source manager to crash.

module Foo;
endmodule
