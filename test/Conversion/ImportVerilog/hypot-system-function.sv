// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s | FileCheck %s --check-prefix=MOORE
// REQUIRES: slang
// UNSUPPORTED: valgrind

// CHECK-LABEL: moore.module @HypotSystemFunction
// CHECK: moore.builtin.hypot {{.*}}, {{.*}} : f64
// MOORE-LABEL: moore.module @HypotSystemFunction
// MOORE: moore.builtin.hypot {{.*}}, {{.*}} : f64
module HypotSystemFunction
    (input real a,
     input real b,
     output real out);
  initial begin
    out = $hypot(a, b);
  end
endmodule
