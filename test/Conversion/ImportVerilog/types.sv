// RUN: circt-translate --import-verilog %s

// CHECK-LABEL: moore.module @IntAtoms
module IntAtoms;
  // CHECK-NEXT: %d0 = moore.variable : !moore.logic
  // CHECK-NEXT: %d1 = moore.variable : !moore.bit
  // CHECK-NEXT: %d2 = moore.variable : !moore.reg
  logic d0;
  bit d1;
  reg d2;

  // CHECK-NEXT: %u0 = moore.variable : !moore.logic
  // CHECK-NEXT: %u1 = moore.variable : !moore.bit
  // CHECK-NEXT: %u2 = moore.variable : !moore.reg
  logic unsigned u0;
  bit unsigned u1;
  reg unsigned u2;

  // CHECK-NEXT: %s0 = moore.variable : !moore.logic<signed>
  // CHECK-NEXT: %s1 = moore.variable : !moore.bit<signed>
  // CHECK-NEXT: %s2 = moore.variable : !moore.reg<signed>
  logic signed s0;
  bit signed s1;
  reg signed s2;
endmodule

// CHECK-LABEL: moore.module @PackedRangeDim
module PackedRangeDim;
  // CHECK-NEXT: %d0 = moore.variable : !moore.packed<range<logic, 2:0>>
  // CHECK-NEXT: %d1 = moore.variable : !moore.packed<range<logic, 0:2>>
  logic [2:0] d0;
  logic [0:2] d1;
endmodule

// CHECK-LABEL: moore.module @MultiPackedRangeDim
module MultiPackedRangeDim;
  // CHECK-NEXT: %v0 = moore.variable : !moore.packed<range<range<logic, 2:0>, 5:0>>
  // CHECK-NEXT: %v1 = moore.variable : !moore.packed<range<range<logic, 2:0>, 5:0>>
  logic [5:0][2:0] v0;
  logic [0:5][2:0] v1;
endmodule
