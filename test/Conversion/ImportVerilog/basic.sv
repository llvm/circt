// RUN: circt-translate --import-verilog %s | FileCheck %s
// REQUIRES: slang

// CHECK: module {
// CHECK: }
module Foo;
endmodule
