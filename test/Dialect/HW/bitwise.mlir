// RUN: circt-opt %s | FileCheck %s

// CHECK-LABEL: func @bitwise(%arg0: i7, %arg1: i7) -> i21 {
func @bitwise(%a: i7, %b: i7) -> i21 {
// CHECK-NEXT:    [[RES0:%[0-9]+]] = comb.xor %arg0 : i7
// CHECK-NEXT:    [[RES1:%[0-9]+]] = comb.or  %arg0, %arg1 : i7
// CHECK-NEXT:    [[RES2:%[0-9]+]] = comb.and %arg0, %arg1, %arg0 : i7
  %and1 = comb.xor %a : i7
  %or1  = comb.or  %a, %b : i7
  %xor1 = comb.and %a, %b, %a : i7

// CHECK-NEXT:    [[RESULT:%[0-9]+]] = comb.concat [[RES0]], [[RES1]], [[RES2]] : (i7, i7, i7) -> i21
// CHECK-NEXT:    return [[RESULT]] : i21
  %result = comb.concat %and1, %or1, %xor1 : (i7, i7, i7) -> i21
  return %result : i21
}


// CHECK-LABEL: func @shl_op(%arg0: i7, %arg1: i7) -> i7 {
func @shl_op(%a: i7, %b: i7) -> i7 {
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.shl  %arg0, %arg1 : i7
  %0  = comb.shl  %a, %b : i7
// CHECK-NEXT:    return [[RES]]
  return %0 : i7
}

// CHECK-LABEL: func @shr_op(%arg0: i7, %arg1: i7) -> i7 {
func @shr_op(%a: i7, %b: i7) -> i7 {
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.shru %arg0, %arg1 : i7
  %0  = comb.shru %a, %b : i7
// CHECK-NEXT:    return [[RES]]
  return %0 : i7
}

// CHECK-LABEL: func @casts(%arg0: i7) -> !hw.struct<int: i7>
func @casts(%in1: i7) -> !hw.struct<int: i7> {
  // CHECK-NEXT: %0 = hw.bitcast %arg0 : (i7) -> !hw.array<7xi1>
  // CHECK-NEXT: %1 = hw.bitcast %0 : (!hw.array<7xi1>) -> !hw.struct<int: i7>
  %bits = hw.bitcast %in1 : (i7) -> !hw.array<7xi1>
  %backToInt = hw.bitcast %bits : (!hw.array<7xi1>) -> !hw.struct<int: i7>
  return %backToInt : !hw.struct<int: i7>
}
