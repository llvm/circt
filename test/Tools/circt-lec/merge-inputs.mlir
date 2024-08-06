// RUN: circt-lec %S/Inputs/a.mlir %S/Inputs/b.mlir --c1 top_a --c2 top_b --emit-mlir | FileCheck %s

// Check constants to make sure a.mlir and b.mlir are properly merged.
// CHECK-LABEL: func.func @foo_0(%arg0: !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT: %c2_bv8 = smt.bv.constant #smt.bv<2>
// CHECK-NEXT: %0 = smt.bv.add %arg0, %c2_bv8
// CHECK-NEXT: return %0

// CHECK-LABEL: func.func @foo(%arg0: !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT: %c1_bv8 = smt.bv.constant #smt.bv<1>
// CHECK-NEXT: %0 = smt.bv.add %arg0, %c1_bv8
// CHECK-NEXT: return %0

// CHECK-LABEL: func.func @top_a
// CHECK:      %[[RESULT1:.+]] = func.call @foo(%[[ARG:.+]])
// CHECK-NEXT: %[[RESULT2:.+]] = func.call @foo_0(%[[ARG]])
// CHECK-NEXT: %[[VAL:.+]] = smt.distinct %[[RESULT1]], %[[RESULT2]]
// CHECK-NEXT: smt.assert %[[VAL]]
