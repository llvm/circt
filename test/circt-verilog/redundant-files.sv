// RUN: circt-verilog %s %s
// REQUIRES: slang
// Internal issue in Slang v9 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

// Listing a file twice caused the Slang's source manager to crash.

module Foo;
endmodule
