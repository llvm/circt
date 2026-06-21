// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

// CHECK-LABEL: func.func @f64_to_i32
// CHECK-SAME: (%arg0: f64) -> i32
func.func @f64_to_i32(%input: !moore.f64) -> !moore.i32 {
  // CHECK: arith.fptosi %arg0 : f64 to i32
  %result = moore.conversion %input : !moore.f64 -> !moore.i32
  return %result : !moore.i32
}

// CHECK-LABEL: func.func @f32_to_i64
// CHECK-SAME: (%arg0: f32) -> i64
func.func @f32_to_i64(%input: !moore.f32) -> !moore.i64 {
  // CHECK: arith.fptosi %arg0 : f32 to i64
  %result = moore.conversion %input : !moore.f32 -> !moore.i64
  return %result : !moore.i64
}
