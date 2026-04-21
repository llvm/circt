// RUN: circt-translate --import-verilog %s -mlir-print-debuginfo | FileCheck %s
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

// CHECK: moore.module @RangeTest
module RangeTest;
  // CHECK: %a = moore.variable : <i32> loc(#[[VAR_LOC:loc[0-9]+]])
  int a;
  // CHECK: moore.procedure initial {
  initial begin
    // CHECK: %[[CONST:.*]] = moore.constant 1 : i32 loc(#[[CONST_LOC:loc[0-9]+]])
    // CHECK: moore.blocking_assign %a, %[[CONST]] : i32 loc(#[[ASSIGN_LOC:loc[0-9]+]])
    a = 1;
  end
endmodule

// CHECK-DAG: #[[VAR_LOC]] = loc("{{.*}}":10:7)
// CHECK-DAG: #[[CONST_LOC]] = loc("{{.*}}":15:9 to :10)
// CHECK-DAG: #[[ASSIGN_LOC]] = loc("{{.*}}":15:5 to :10)
