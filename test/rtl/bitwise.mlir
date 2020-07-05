// RUN: circt-opt %s | FileCheck %s

// CHECK-LABEL: func @bitwise(%arg0: i7, %arg1: i7) -> i21 {
func @bitwise(%a: i7, %b: i7) -> i21 {
// CHECK-NEXT:    %0 = rtl.and %arg0, %arg1 : i7
// CHECK-NEXT:    %1 = rtl.or  %arg0, %arg1 : i7
// CHECK-NEXT:    %2 = rtl.xor %arg0, %arg1 : i7
  %and1 = rtl.and %a, %b : i7
  %or1  = rtl.or  %a, %b : i7
  %xor1 = rtl.xor %a, %b : i7

// CHECK-NEXT:    %[[RESULT:.*]] = rtl.concat %0, %1, %2 : (i7, i7, i7) -> i21
// CHECK-NEXT:    return %[[RESULT]] : i21
  %result = rtl.concat %and1, %or1, %xor1 : (i7, i7, i7) -> i21
  return %result : i21
}
