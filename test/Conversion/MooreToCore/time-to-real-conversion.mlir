// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

// CHECK-LABEL: func.func @time_to_f64
// CHECK-SAME: (%arg0: !llhd.time) -> f64
func.func @time_to_f64(%input: !moore.time) -> !moore.f64 {
  // CHECK: %[[INT:.*]] = llhd.time_to_int %arg0
  // CHECK: %[[REAL:.*]] = arith.uitofp %[[INT]] : i64 to f64
  %result = moore.conversion %input : !moore.time -> !moore.f64
  // CHECK: return %[[REAL]] : f64
  return %result : !moore.f64
}

// CHECK-LABEL: func.func @time_to_f32
// CHECK-SAME: (%arg0: !llhd.time) -> f32
func.func @time_to_f32(%input: !moore.time) -> !moore.f32 {
  // CHECK: %[[INT:.*]] = llhd.time_to_int %arg0
  // CHECK: %[[REAL:.*]] = arith.uitofp %[[INT]] : i64 to f32
  %result = moore.conversion %input : !moore.time -> !moore.f32
  // CHECK: return %[[REAL]] : f32
  return %result : !moore.f32
}
