// RUN: circt-opt %s | FileCheck %s

// CHECK-LABEL: func @test1(%arg0: i3) -> i50 {
func @test1(%arg0: i3) -> i50 {
  // CHECK-NEXT:    %c42_i12 = rtl.constant(42 : i12) : i12
  // CHECK-NEXT:    %[[RES0:.*]] = rtl.add %c42_i12, %c42_i12 : i12
  // CHECK-NEXT:    %[[RES1:.*]] = rtl.mul %c42_i12, %[[RES0:.*]] : i12
  %a = rtl.constant(42 : i12) : i12
  %b = rtl.add %a, %a : i12
  %c = rtl.mul %a, %b : i12

  // CHECK-NEXT:    %[[RES2:.*]] = rtl.sext %arg0 : (i3) -> i7
  // CHECK-NEXT:    %[[RES3:.*]] = rtl.zext %arg0 : (i3) -> i7
  %d = rtl.sext %arg0 : (i3) -> i7
  %e = rtl.zext %arg0 : (i3) -> i7

  // CHECK-NEXT:    %[[RES4:.*]] = rtl.concat %[[C42_i12:.*]] : (i12) -> i12
  %conc1 = rtl.concat %a : (i12) -> i12

  // CHECK-NEXT:    %[[RES5:.*]] = rtl.andr %[[RES4:.*]] : i12
  // CHECK-NEXT:    %[[RES6:.*]] = rtl.orr  %[[RES4:.*]] : i12
  // CHECK-NEXT:    %[[RES7:.*]] = rtl.xorr %[[RES4:.*]] : i12
  %andr1 = rtl.andr %conc1 : i12
  %orr1  = rtl.orr  %conc1 : i12
  %xorr1 = rtl.xorr %conc1 : i12

  // CHECK-NEXT:    %[[RES8:.*]] = rtl.concat %[[RES4:.*]], %[[RES0:.*]], %[[RES1:.*]], %[[RES2:.*]], %[[RES3:.*]] : (i12, i12, i12, i7, i7) -> i50
  %result = rtl.concat %conc1, %b, %c, %d, %e : (i12, i12, i12, i7, i7) -> i50

  // CHECK-NEXT: %[[RES9:.*]] = rtl.extract [[RES8:.*]] from 4 : (i50) -> i19
  %small1 = rtl.extract %result from 4 : (i50) -> i19

  // CHECK-NEXT: %[[RES10:.*]] = rtl.extract [[RES8:.*]] from 31 : (i50) -> i19
  %small2 = rtl.extract %result from 31 : (i50) -> i19

  // CHECK-NEXT: rtl.add %[[RES9:.*]], %[[RES10:.*]] : i19
  %add = rtl.add %small1, %small2 : i19

  // CHECK-NEXT:  = rtl.wire : i4
  %w = rtl.wire : i4

  // CHECK-NEXT:    return [[RES8:.*]] : i50
  return %result : i50
}
// CHECK-NEXT:  }