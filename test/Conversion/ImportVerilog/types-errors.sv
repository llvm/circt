// RUN: circt-translate --import-verilog --verify-diagnostics --split-input-file %s
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

module Foo;
  // expected-error @below {{unsupported type}}
  // expected-note @below {{}}
  union { bit a; logic b; } x;
endmodule
