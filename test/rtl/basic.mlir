// RUN: cirt-opt %s | FileCheck %s

func @test1() -> i36 {
  %a = rtl.constant(42 : i12) : i12
  %b = rtl.add %a, %a : i12
  %c = rtl.mul %a, %b : i12
  %d = rtl.concat %a, %b, %c : (i12, i12, i12) -> i36
  return %d : i36
}

// CHECK-LABEL: func @test1() -> i36 {
// CHECK-NEXT:    %c42_i12 = rtl.constant(42 : i12) : i12
// CHECK-NEXT:    %0 = rtl.add %c42_i12, %c42_i12 : i12
// CHECK-NEXT:    %1 = rtl.mul %c42_i12, %0 : i12
// CHECK-NEXT:    %2 = rtl.concat %c42_i12, %0, %1 : (i12, i12, i12) -> i36
// CHECK-NEXT:    return %2 : i36
// CHECK-NEXT:  }
