// RUN: circt-translate --export-aiger %s -o %t.aig --verify-diagnostics
// RUN: cat %t.aig | FileCheck %s --check-prefix=BINARY
// RUN: circt-translate --import-aiger %t.aig | FileCheck %s

// Test complex binary encoding with proper ordering
// BINARY:      aig 9 3 2 2 4
hw.module @mixed_logic(in %a: i1, in %b: i1, in %c: i1, out x: i1, out y: i1, in %clk: !seq.clock) {
  %comb_logic = aig.and_inv %a, %b : i1
  %next_x = aig.and_inv %c, %comb_logic : i1
  %next_y = aig.and_inv %reg_x, %reg_y : i1
  %reg_x = seq.compreg %next_x, %clk : i1
  %reg_y = seq.compreg %next_y, %clk : i1
  %final_out = aig.and_inv %next_x, %next_y : i1
  hw.output %final_out, %reg_y : i1, i1
}

// CHECK-LABEL:  hw.module @aiger_top(in %a : i1, in %b : i1, in %c : i1, out x : i1, out y : i1, in %clock : !seq.clock) {
// CHECK:         %reg_x = seq.compreg %[[AND_INV_2:.+]], %clock : i1  
// CHECK-NEXT:    %reg_y = seq.compreg %[[AND_INV_3:.+]], %clock : i1  
// CHECK-NEXT:    %[[AND_INV_1:.+]] = aig.and_inv %b, %a : i1
// CHECK-NEXT:    %[[AND_INV_2]] = aig.and_inv %[[AND_INV_1]], %c : i1
// CHECK-NEXT:    %[[AND_INV_3:.+]] = aig.and_inv %reg_y, %reg_x : i1
// CHECK-NEXT:    %[[AND_INV_4:.+]] = aig.and_inv %[[AND_INV_3]], %[[AND_INV_2]] : i1
// CHECK-NEXT:    hw.output %[[AND_INV_4]], %reg_y : i1, i1
// CHECK-NEXT: }
