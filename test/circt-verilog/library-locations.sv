// RUN: circt-verilog -y %S/include --mlir-print-debuginfo %s | FileCheck %s
// REQUIRES: slang
// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

// See https://github.com/llvm/circt/pull/8840

// CHECK-LABEL: hw.module @foo
module foo;
  // CHECK-NEXT: hw.instance
  // CHECK-NEXT: hw.output
  // CHECK-NEXT: } loc([[LOC1:#.+]])
  library_module bar();
endmodule

// CHECK-LABEL: hw.module private @library_module
// CHECK-NEXT: hw.output
// CHECK-NEXT: } loc([[LOC2:#.+]])

// CHECK: [[LOC1]] = loc("{{.*}}library-locations.sv"
// CHECK: [[LOC2]] = loc("{{.*}}include{{/|\\\\}}library_module.sv"
