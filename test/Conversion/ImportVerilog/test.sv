// RUN: circt-translate --import-verilog %s | FileCheck %s
// REQUIRES: slang

// CHECK-LABEL: moore.module @Port
// CHECK-NEXT: moore.port In "a"
// CHECK-NEXT: %a = moore.net  "wire" : !moore.packed<range<logic, 3:0>>
// CHECK-NEXT: moore.port Out "b"
// CHECK-NEXT: %b = moore.net  "wire" : !moore.packed<range<logic, 3:0>>
module Port(input [3:0] a, output [3:0] b);
endmodule
