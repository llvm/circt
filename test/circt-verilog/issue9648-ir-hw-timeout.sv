// RUN: circt-verilog --ir-hw %s | FileCheck %s
// REQUIRES: slang

module M(output logic O);
  typedef struct packed { logic a; logic b; } S;
  S s;

  always_comb s.b = 0;
  assign O = s.a;

  // CHECK-LABEL: hw.module @M(
  // CHECK: hw.struct_inject
  // CHECK: hw.output
endmodule
