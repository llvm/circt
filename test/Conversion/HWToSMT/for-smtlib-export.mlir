// RUN: circt-opt %s --convert-hw-to-smt='for-smtlib-export' --split-input-file | FileCheck %s

// CHECK-LABEL: smt.solver() : () -> ()
// CHECK-NEXT: %[[DECL1:.+]] = smt.declare_fun : !smt.bv<32>
// CHECK-NEXT: %[[DECL2:.+]] = smt.declare_fun : !smt.bv<32>
// CHECK-NEXT: %[[EQ:.+]] = smt.eq %[[DECL1]], %[[DECL2]] : !smt.bv<32>
// CHECK-NEXT: smt.assert %[[EQ]]

hw.module @modA(in %in: i32, out out: i32) {
  hw.output %in : i32
}

// -----

// Check that output assertions are generated correctly with multiple outputs
// and make sure input/output decls are distinct
// CHECK-LABEL: smt.solver() : () -> ()
// CHECK-NEXT: %[[ARG0:.*]] = smt.declare_fun : !smt.bv<16>
// CHECK-NEXT: %[[CONST0:.*]] = smt.bv.constant #smt.bv<42> : !smt.bv<32>
// CHECK-NEXT: %[[CONST1:.*]] = smt.bv.constant #smt.bv<43> : !smt.bv<32>
// CHECK-NEXT: %[[DECL1:.*]] = smt.declare_fun : !smt.bv<32>
// CHECK-NEXT: %[[EQ1:.*]] = smt.eq %[[CONST0]], %[[DECL1]] : !smt.bv<32>
// CHECK-NEXT: smt.assert %[[EQ1]]
// CHECK-NEXT: %[[DECL2:.*]] = smt.declare_fun : !smt.bv<32>
// CHECK-NEXT: %[[EQ2:.*]] = smt.eq %[[CONST1]], %[[DECL2]] : !smt.bv<32>
// CHECK-NEXT: smt.assert %[[EQ2]]

hw.module @modB(in %in1: i16, out out1: i32, out out2: i32) {
  %const1 = hw.constant 42 : i32
  %const2 = hw.constant 43 : i32
  hw.output %const1, %const2 : i32, i32
}

