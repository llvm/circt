// RUN: circt-opt %s -export-verilog | FileCheck %s

module attributes {circt.loweringOptions = "locationInfoStyle=wrapInAtSquareBracket"}{
// CHECK: module Foo();
// CHECK-SAME: // @[dummy:1:1]
// CHECK-NEXT: endmodule
hw.module @Foo() -> () {
  hw.output
} loc("dummy":1:1)
}