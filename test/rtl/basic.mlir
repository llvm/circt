// RUN: circt-opt %s | FileCheck %s

// CHECK-LABEL: func @test1(%arg0: i3) -> i50 {
func @test1(%arg0: i3) -> i50 {
  // CHECK-NEXT:    %c42_i12 = rtl.constant(42 : i12) : i12
  // CHECK-NEXT:    %0 = rtl.add %c42_i12, %c42_i12 : i12
  // CHECK-NEXT:    %1 = rtl.mul %c42_i12, %0 : i12
  %a = rtl.constant(42 : i12) : i12
  %b = rtl.add %a, %a : i12
  %c = rtl.mul %a, %b : i12

  // CHECK-NEXT:    %2 = rtl.sext %arg0 : i3, i7
  // CHECK-NEXT:    %3 = rtl.zext %arg0 : i3, i7
  %d = rtl.sext %arg0 : i3, i7
  %e = rtl.zext %arg0 : i3, i7

  // CHECK-NEXT:    %4 = rtl.concat %c42_i12 : (i12) -> i12
  %conc1 = rtl.concat %a : (i12) -> i12

  // CHECK-NEXT:    %5 = rtl.andr %4 : i12
  // CHECK-NEXT:    %6 = rtl.orr  %4 : i12
  // CHECK-NEXT:    %7 = rtl.xorr %4 : i12
  %andr1 = rtl.andr %conc1 : i12
  %orr1  = rtl.orr  %conc1 : i12
  %xorr1 = rtl.xorr %conc1 : i12

  // CHECK-NEXT:    %8 = rtl.concat %4, %0, %1, %2, %3 : (i12, i12, i12, i7, i7) -> i50
  %result = rtl.concat %conc1, %b, %c, %d, %e : (i12, i12, i12, i7, i7) -> i50
  // CHECK-NEXT:    return %8 : i50
  return %result : i50
}
// CHECK-NEXT:  }
