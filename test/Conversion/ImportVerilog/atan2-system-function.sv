// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s | FileCheck %s
// REQUIRES: slang
// UNSUPPORTED: valgrind

// CHECK-LABEL: moore.module @Atan2SystemFunction
module Atan2SystemFunction(
  input real y,
  input real x,
  output real out
);
  initial out = $atan2(y, x);
endmodule

// CHECK: moore.builtin.atan2 {{.*}}, {{.*}} : f64
// CHECK-NOT: moore.conditional
