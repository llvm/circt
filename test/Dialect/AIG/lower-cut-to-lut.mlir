// RUN: circt-opt %s --aig-lower-cut-to-lut | FileCheck %s
// CHECK: hw.module @Cut
hw.module @Cut(in %a: i1, in %b: i1, in %c: i1, in %d: i1, out e: i1) {
  // CHECK-NEXT:  %0 = comb.truth_table %a, %b, %c, %d
  // CHECK-SAME:       -> [false, false, false, false, false, true, true, true,
  // CHECK-SAME:           false, false, false, false, false, false, false, false]
  %0 = aig.cut %a, %b, %c, %d : (i1, i1, i1, i1) -> (i1) {
  ^bb0(%arg0: i1, %arg1: i1, %arg2: i1, %arg3: i1):
    %1 = aig.and_inv not %arg0, not %arg1 : i1
    %2 = aig.and_inv %arg2, not %arg3 : i1
    %3 = aig.and_inv not %1, %2 : i1
    aig.output %3 : i1
  }
  hw.output %0 : i1
}
