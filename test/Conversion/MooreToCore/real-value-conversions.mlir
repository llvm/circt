// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

// CHECK-LABEL: func.func @f32_to_f64
// CHECK-SAME: (%arg0: f32) -> f64
func.func @f32_to_f64(%input: !moore.f32) -> !moore.f64 {
  // CHECK: arith.extf %arg0 : f32 to f64
  %result = moore.conversion %input : !moore.f32 -> !moore.f64
  return %result : !moore.f64
}

// CHECK-LABEL: func.func @f64_to_f32
// CHECK-SAME: (%arg0: f64) -> f32
func.func @f64_to_f32(%input: !moore.f64) -> !moore.f32 {
  // CHECK: arith.truncf %arg0 : f64 to f32
  %result = moore.conversion %input : !moore.f64 -> !moore.f32
  return %result : !moore.f32
}
