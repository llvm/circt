// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

// ImportVerilog treats a case statement as exhaustive, and turns its last item
// into the default, when the constant labels cover every two-state value of the
// selector. This used to be decided by counting labels, so repeated labels
// (within one item or across items) made a case with uncovered values look
// full: the default was dropped and the last item ran for the uncovered values
// instead. These tests pin that exhaustiveness depends on the set of distinct
// label values, that a repeated label keeps first-match priority, and that a
// really exhaustive case still drops its default.

// 8 labels on a 3-bit selector but only 6 distinct values (6 and 4 repeat);
// values 0 and 2 are uncovered, so the default (z = 12) must be generated and
// reached when the last label does not match. Labels are checked in order, so
// the first `3'd4` (second item) is the one that matches value 4.
// CHECK-LABEL: @dupLabelsNotExhaustive
function void dupLabelsNotExhaustive(logic [2:0] a);
  logic [3:0] z;
  case (a)
    // CHECK: moore.constant 3 : i3
    // CHECK: moore.constant 1 : i3
    3'd3, 3'd1: z = 4'd1;
    // CHECK: moore.constant -1 : i3
    // CHECK: moore.constant -4 : i3
    3'd7, 3'd4: z = 4'd2;
    // CHECK: moore.constant -3 : i3
    3'd5: z = 4'd4;
    // CHECK: moore.constant -2 : i3
    // CHECK: moore.constant -2 : i3
    3'd6, 3'd6: z = 4'd8;
    // The last label's compare branches to the last item or to the default.
    // CHECK: moore.constant -4 : i3
    // CHECK: cf.cond_br {{%.+}}, [[LAST:\^.+]], [[DEFAULT:\^.+]]
    // CHECK: [[LAST]]:
    // CHECK: moore.constant 3 : i4
    3'd4: z = 4'd3;
    // CHECK: [[DEFAULT]]:
    // CHECK-NEXT: moore.constant -4 : i4
    default: z = 4'd12;
  endcase
endfunction

// Four labels on a 2-bit selector, one repeated across items, and value 0
// uncovered. Counting labels said "exhaustive"; the default (z = 13) must stay.
// CHECK-LABEL: @dupLabelAcrossItemsNotExhaustive
function void dupLabelAcrossItemsNotExhaustive(logic [1:0] a);
  logic [3:0] z;
  case (a)
    // CHECK: moore.constant 1 : i2
    2'd1: z = 4'd1;
    // CHECK: moore.constant 1 : i2
    2'd1: z = 4'd2;
    // CHECK: moore.constant -2 : i2
    2'd2: z = 4'd3;
    // CHECK: moore.constant -1 : i2
    // CHECK: cf.cond_br {{%.+}}, [[LAST:\^.+]], [[DEFAULT:\^.+]]
    // CHECK: [[LAST]]:
    // CHECK: moore.constant 4 : i4
    2'd3: z = 4'd4;
    // CHECK: [[DEFAULT]]:
    // CHECK-NEXT: moore.constant -3 : i4
    default: z = 4'd13;
  endcase
endfunction

// Every value of the selector is present (and one is repeated, which never
// matches): this is exhaustive, so the default is dropped and the final label
// mismatch jumps straight to the last item.
// CHECK-LABEL: @dupLabelExhaustive
function void dupLabelExhaustive(logic [1:0] a);
  logic [3:0] z;
  case (a)
    // CHECK: moore.constant 0 : i2
    2'd0: z = 4'd1;
    // CHECK: moore.constant 1 : i2
    2'd1: z = 4'd2;
    // CHECK: moore.constant -2 : i2
    2'd2: z = 4'd3;
    // CHECK: moore.constant -1 : i2
    2'd3: z = 4'd4;
    // CHECK: moore.constant 1 : i2
    // CHECK: cf.cond_br {{%.+}}, [[LAST:\^.+]], [[ELSE:\^.+]]
    // CHECK: [[LAST]]:
    // CHECK: moore.constant 5 : i4
    2'd1: z = 4'd5;
    // CHECK: [[ELSE]]:
    // CHECK-NOT: moore.constant -3 : i4
    // CHECK-NEXT: cf.br [[LAST]]
    default: z = 4'd13;
  endcase
endfunction
