// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang
// UNSUPPORTED: valgrind

// Element and range selects into packed structs and unions are bit selects
// into the equivalent vector of bits (IEEE 1800-2017 § 7.2.1).

typedef struct packed { logic [2:0] a; logic [1:0] b; } my_struct_t;
typedef union packed { logic [4:0] bits; my_struct_t fields; } my_union_t;

// CHECK-LABEL: moore.module @PackedAggregateSelect
module PackedAggregateSelect;
  my_struct_t s;
  my_union_t u;
  logic [2:0] i;
  logic x;
  logic [1:0] y;
  initial begin
    // Rvalue constant bit select: convert to the simple bit vector, then a
    // plain extract.
    // CHECK: [[SBV0:%.+]] = moore.packed_to_sbv
    // CHECK: moore.extract [[SBV0]] from 4
    x = s[4];
    // Rvalue dynamic bit select.
    // CHECK: [[SBV1:%.+]] = moore.packed_to_sbv
    // CHECK: moore.dyn_extract [[SBV1]] from
    x = s[i];
    // Rvalue range select (part select).
    // CHECK: [[SBV2:%.+]] = moore.packed_to_sbv
    // CHECK: moore.extract [[SBV2]] from 1
    y = s[2:1];
    // Lvalue constant bit select: the ref-typed extract handles the packed
    // aggregate directly.
    // CHECK: moore.extract_ref %s from 0
    s[0] = 1'b1;
    // Lvalue dynamic bit select.
    // CHECK: moore.dyn_extract_ref %s from
    s[i] = 1'b0;
    // Packed union rvalue and lvalue selects.
    // CHECK: [[SBV3:%.+]] = moore.packed_to_sbv
    // CHECK: moore.extract [[SBV3]] from 2
    x = u[2];
    // CHECK: moore.dyn_extract_ref %u from
    u[i] = 1'b1;
  end
endmodule
