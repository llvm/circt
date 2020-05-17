// RUN: cirt-opt %s | FileCheck %s

func @test1() -> i12 {
  %a = rtl.constant(42 : i12) : i12
  %b = rtl.add %a, %a : i12
  %c = rtl.mul %a, %b : i12
  return %c : i12
}

// CHECK-LABEL: func @test1() -> i12 {
// CHECK-NEXT:    %0 = rtl.constant(42 : i12) : i12
// CHECK-NEXT:    %1 = rtl.add %0, %0 : i12
// CHECK-NEXT:    %2 = rtl.mul %0, %1 : i12
// CHECK-NEXT:    return %2 : i12
// CHECK-NEXT:  }
