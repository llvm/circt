// RUN: circt-opt %s -export-verilog --split-input-file | FileCheck %s

module attributes {circt.loweringOptions = "locationInfoStyle=wrapInAtSquareBracket"}{
// CHECK: module Foo();
// CHECK-SAME: // @[dummy:1:1]
// CHECK-NEXT: endmodule
hw.module @Foo() -> () {
  hw.output
} loc("dummy":1:1)
}

// -----

module attributes {circt.loweringOptions = "locationInfoStyle=plain"}{
// CHECK: module Foo();
// CHECK-SAME: // dummy:1:1
// CHECK-NEXT: endmodule
hw.module @Foo() -> () {
  hw.output
} loc("dummy":1:1)
}

// -----

module attributes {circt.loweringOptions = "locationInfoStyle=none"}{
// CHECK: module Foo();
// CHECK-NOT: //
// CHECK-NEXT: endmodule
hw.module @Foo() -> () {
  hw.output
} loc("dummy":1:1)
}
