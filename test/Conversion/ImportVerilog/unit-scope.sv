// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang
//
// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

// Compilation-unit ($unit) variable and function declarations.
int a = 5;

function automatic int add1
    (int x);
  return x + 1;
endfunction

// CHECK: moore.global_variable @a : !moore.i32
// CHECK: func.func private @add1

// CHECK-LABEL: moore.module @use_unit
module use_unit;
  int b;
  initial begin
    // CHECK: [[B:%.*]] = moore.variable : <i32>
    // CHECK: [[GA:%.*]] = moore.get_global_variable @a
    // CHECK: [[AV:%.*]] = moore.read [[GA]] : <i32>
    // CHECK: moore.blocking_assign [[B]], [[AV]]
    b = a;
    // CHECK: [[BV:%.*]] = moore.read [[B]] : <i32>
    // CHECK: [[SUM:%.*]] = func.call @add1([[BV]])
    // CHECK: moore.blocking_assign [[B]], [[SUM]]
    b = add1(b);
  end
endmodule
