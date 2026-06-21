// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s | FileCheck %s
// REQUIRES: slang
// UNSUPPORTED: valgrind

// CHECK-LABEL: moore.module @RealInsideScalar
module RealInsideScalar
    (input real r,
     output logic eq,
     output logic range);
  initial begin
    // CHECK: moore.feq
    eq = r inside {1.0, 2.5};
    // CHECK: moore.fge
    // CHECK: moore.fle
    // CHECK: moore.and
    range = r inside {[3.0 : 4.0]};
  end
endmodule

// CHECK-LABEL: moore.module @StringInsideScalar
module StringInsideScalar
    (output logic found);
  string s;

  initial begin
    s = "hi";
    // CHECK: moore.string_cmp eq
    // CHECK: moore.string_cmp eq
    // CHECK: moore.or
    found = s inside {"hi", "bye"};
  end
endmodule
