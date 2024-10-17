// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK-LABEL: @And
// CHECK-NEXT: aig.and_inv %b, %b : i4
// CHECK-NEXT: aig.and_inv %b, not %b : i4
// CHECK-NEXT: aig.and_inv not %a, not %a : i1
hw.module @And(in %a: i1, in %b: i4) {
  %0 = aig.and_inv %b, %b : i4
  %1 = aig.and_inv %b, not %b : i4
  %2 = aig.and_inv not %a, not %a : i1
}

hw.module @Cut(in %a: i1, in %b: i1, out c: i1, out d: i1) {
  %0, %1 = aig.cut %a, %b : (i1, i1) -> (i1, i1) {
  ^bb0(%arg0: i1, %arg1: i1):
    %c = aig.and_inv %arg0, not %arg1 : i1
    %d = aig.and_inv %arg0, %arg1 : i1
    aig.output %c, %d : i1, i1
  }
  hw.output %0, %1 : i1, i1
}
