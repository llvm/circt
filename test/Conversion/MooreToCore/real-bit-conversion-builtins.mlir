// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

// CHECK-LABEL: func.func @RealToBits
// CHECK-SAME: (%arg0: f64) -> i64
func.func @RealToBits(%value: !moore.f64) -> !moore.i64 {
  // CHECK: %[[BITS:.*]] = arith.bitcast %arg0 : f64 to i64
  %bits = moore.builtin.realtobits %value
  // CHECK: return %[[BITS]] : i64
  return %bits : !moore.i64
}

// CHECK-LABEL: func.func @BitsToReal
// CHECK-SAME: (%arg0: i64) -> f64
func.func @BitsToReal(%value: !moore.i64) -> !moore.f64 {
  // CHECK: %[[REAL:.*]] = arith.bitcast %arg0 : i64 to f64
  %real = moore.builtin.bitstoreal %value : !moore.i64
  // CHECK: return %[[REAL]] : f64
  return %real : !moore.f64
}

// CHECK-LABEL: func.func @ShortRealToBits
// CHECK-SAME: (%arg0: f32) -> i32
func.func @ShortRealToBits(%value: !moore.f32) -> !moore.i32 {
  // CHECK: %[[BITS:.*]] = arith.bitcast %arg0 : f32 to i32
  %bits = moore.builtin.shortrealtobits %value
  // CHECK: return %[[BITS]] : i32
  return %bits : !moore.i32
}

// CHECK-LABEL: func.func @BitsToShortReal
// CHECK-SAME: (%arg0: i32) -> f32
func.func @BitsToShortReal(%value: !moore.i32) -> !moore.f32 {
  // CHECK: %[[REAL:.*]] = arith.bitcast %arg0 : i32 to f32
  %real = moore.builtin.bitstoshortreal %value : !moore.i32
  // CHECK: return %[[REAL]] : f32
  return %real : !moore.f32
}
