// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang
// UNSUPPORTED: valgrind

// Case comparisons over packed aggregates use the vector-of-bits
// equivalent (IEEE 1800-2017 § 7.2.1).

typedef union packed { logic [5:0] a; logic [5:0] b; } op_u;
typedef struct packed { op_u t; } op_s;

// CHECK-LABEL: moore.module @CasePackedAggregate
module CasePackedAggregate;
  op_s op;
  logic r;
  always_comb begin
    // CHECK: [[SBV:%.+]] = moore.packed_to_sbv
    // CHECK: moore.case_eq
    // CHECK: moore.case_eq
    unique case (op)
      6'd1, 6'd2: r = 1'b0;
      default:    r = 1'b1;
    endcase
  end
endmodule
