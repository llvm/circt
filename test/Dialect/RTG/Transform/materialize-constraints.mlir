// RUN: circt-opt %s --rtg-materialize-constraints | FileCheck %s

// CHECK-LABEL: @test
rtg.test @test() {
  // CHECK-NEXT: rtgtest.implicit_constraint_op{{$}}
  // CHECK-NEXT: [[V0:%.+]] = rtg.constant true
  // CHECK-NEXT: rtg.constraint [[V0]]
  rtgtest.implicit_constraint_op implicit_constraint
  // CHECK-NEXT: rtgtest.implicit_constraint_op{{$}}
  rtgtest.implicit_constraint_op
  // CHECK-NEXT: }
}
