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

// CHECK: moore.global_variable @__unit__a : !moore.i32
// CHECK: func.func private @__unit__add1

// CHECK-LABEL: moore.module @use_unit
module use_unit;
  int b;
  initial begin
    b = a;
    b = add1(b);
  end
endmodule
