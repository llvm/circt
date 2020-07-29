// RUN: circt-opt %s | FileCheck %s

// CHECK-LABEL: func @bitwise(%arg0: i7, %arg1: i7) -> i21 {
func @bitwise(%a: i7, %b: i7) -> i21 {
// CHECK-NEXT:    %[[RES0:.*]] = rtl.xor %arg0 : i7
// CHECK-NEXT:    %[[RES1:.*]] = rtl.or  %arg0, %arg1 : i7
// CHECK-NEXT:    %[[RES2:.*]] = rtl.and %arg0, %arg1, %arg0 : i7
  %and1 = rtl.xor %a : i7
  %or1  = rtl.or  %a, %b : i7
  %xor1 = rtl.and %a, %b, %a : i7

// CHECK-NEXT:    %[[RESULT:.*]] = rtl.concat %[[RES0:.*]], %[[RES1:.*]], %[[RES2:.*]] : (i7, i7, i7) -> i21
// CHECK-NEXT:    return %[[RESULT]] : i21
  %result = rtl.concat %and1, %or1, %xor1 : (i7, i7, i7) -> i21
  return %result : i21
}
