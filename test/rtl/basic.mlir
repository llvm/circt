// RUN: cirt-opt %s | FileCheck %s

func @test1() -> i12 {
  %a = rtl.constant(42 : i12) : i12
  return %a : i12
}

// CHECK-LABEL: func @test1() -> i12 {
// CHECK-NEXT:    %0 = rtl.constant(42 : i12) : i12
// CHECK-NEXT:    return %0 : i12
// CHECK-NEXT:  }
