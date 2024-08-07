// RUN: split-file %s %t
// RUN: circt-lec %t/a.mlir %t/b.mlir --c1 top_a --c2 top_b --emit-mlir | FileCheck %s

//--- a.mlir
hw.module @foo(in %a : i8, out b : i8) {
  %c1_i8 = hw.constant 1 : i8
  %add = comb.add %a, %c1_i8: i8
  hw.output %add : i8
}
hw.module @top_a(in %a : i8, out b : i8) {
  %foo.b = hw.instance "foo" @foo(a: %a: i8) -> (b: i8)
  hw.output %foo.b : i8
}

//--- b.mlir
hw.module @foo(in %a : i8, out b : i8) {
  %c2_i8 = hw.constant 2 : i8
  %add = comb.add %a, %c2_i8: i8
  hw.output %add : i8
}

hw.module @top_b(in %a : i8, out b : i8) {
  %foo.b = hw.instance "foo" @foo(a: %a: i8) -> (b: i8)
  hw.output %foo.b : i8
}

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
