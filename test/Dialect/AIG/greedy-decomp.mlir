// RUN: circt-opt %s --aig-greedy-cut-decomp | FileCheck %s

// CHECK-LABEL: @variadic
hw.module @variadic(in %a : i1, in %b : i1, in %c : i1, in %d : i1, in %e : i1, in %f : i1, in %g : i1, out and6 : i1) {
  %0 = aig.and_inv %b, %c : i1
  %1 = aig.and_inv %a, %0 : i1
  %2 = aig.and_inv %d, %e : i1
  %3 = aig.and_inv %f, %g : i1
  %4 = aig.and_inv %2, %3 : i1
  %5 = aig.and_inv %1, %4 : i1
  hw.output %5 : i1
}
// CHECK-NEXT:  %[[RES0:.+]] = aig.cut %a, %b, %c : (i1, i1, i1) -> i1 {
// CHECK-NEXT:  ^bb0(%arg0: i1, %arg1: i1, %arg2: i1):
// CHECK-NEXT:    %[[RES2:.+]] = aig.and_inv %arg1, %arg2 : i1
// CHECK-NEXT:    %[[RES3:.+]] = aig.and_inv %arg0, %[[RES2]] : i1
// CHECK-NEXT:    aig.output %[[RES3]] : i1
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[RES1:.+]] = aig.cut %0, %d, %e, %f, %g : (i1, i1, i1, i1, i1) -> i1 {
// CHECK-NEXT:  ^bb0(%arg0: i1, %arg1: i1, %arg2: i1, %arg3: i1, %arg4: i1):
// CHECK-NEXT:    %[[RES2:.+]] = aig.and_inv %arg1, %arg2 : i1
// CHECK-NEXT:    %[[RES3:.+]] = aig.and_inv %arg3, %arg4 : i1
// CHECK-NEXT:    %[[RES4:.+]] = aig.and_inv %[[RES2]], %[[RES3]] : i1
// CHECK-NEXT:    %[[RES5:.+]] = aig.and_inv %arg0, %[[RES4]] : i1
// CHECK-NEXT:    aig.output %[[RES5]] : i1
// CHECK-NEXT:  }
