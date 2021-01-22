// RUN: circt-opt %s | FileCheck %s

// CHECK-LABEL: func @bitwise(%arg0: i7, %arg1: i7) -> i21 {
func @bitwise(%a: i7, %b: i7) -> i21 {
// CHECK-NEXT:    [[RES0:%[0-9]+]] = rtl.xor %arg0 : i7
// CHECK-NEXT:    [[RES1:%[0-9]+]] = rtl.or  %arg0, %arg1 : i7
// CHECK-NEXT:    [[RES2:%[0-9]+]] = rtl.and %arg0, %arg1, %arg0 : i7
  %and1 = rtl.xor %a : i7
  %or1  = rtl.or  %a, %b : i7
  %xor1 = rtl.and %a, %b, %a : i7

// CHECK-NEXT:    [[RESULT:%[0-9]+]] = rtl.concat [[RES0]], [[RES1]], [[RES2]] : (i7, i7, i7) -> i21
// CHECK-NEXT:    return [[RESULT]] : i21
  %result = rtl.concat %and1, %or1, %xor1 : (i7, i7, i7) -> i21
  return %result : i21
}


// CHECK-LABEL: func @shl_op(%arg0: i7, %arg1: i7) -> i7 {
func @shl_op(%a: i7, %b: i7) -> i7 {
// CHECK-NEXT:    [[RES:%[0-9]+]] = rtl.shl  %arg0, %arg1 : i7
  %0  = rtl.shl  %a, %b : i7
// CHECK-NEXT:    return [[RES]]
  return %0 : i7
}

// CHECK-LABEL: func @shr_op(%arg0: i7, %arg1: i7) -> i7 {
func @shr_op(%a: i7, %b: i7) -> i7 {
// CHECK-NEXT:    [[RES:%[0-9]+]] = rtl.shru %arg0, %arg1 : i7
  %0  = rtl.shru %a, %b : i7
// CHECK-NEXT:    return [[RES]]
  return %0 : i7
}

// CHECK-LABEL: func @casts(%arg0: i7) -> !rtl.struct<int: i7>
func @casts(%in1: i7) -> !rtl.struct<int: i7> {
  // CHECK-NEXT: %0 = rtl.bitcast %arg0 : (i7) -> !rtl.array<7xi1>
  // CHECK-NEXT: %1 = rtl.bitcast %0 : (!rtl.array<7xi1>) -> !rtl.struct<int: i7>
  %bits = rtl.bitcast %in1 : (i7) -> !rtl.array<7xi1>
  %backToInt = rtl.bitcast %bits : (!rtl.array<7xi1>) -> !rtl.struct<int: i7>
  return %backToInt : !rtl.struct<int: i7>
}
