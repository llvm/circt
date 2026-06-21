// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

// CHECK-LABEL: func.func @ZeroWidthRealToInt(
func.func @ZeroWidthRealToInt(%input: !moore.f64) -> !moore.i0 {
  // CHECK: %[[ZERO:.+]] = hw.constant 0 : i0
  // CHECK-NEXT: return %[[ZERO]] : i0
  %0 = moore.real_to_int %input : f64 -> i0
  return %0 : !moore.i0
}

// CHECK-LABEL: func.func @ZeroWidthSIntToReal(
func.func @ZeroWidthSIntToReal(%input: !moore.i0) -> !moore.f64 {
  // CHECK: %[[ZERO:.+]] = arith.constant 0.000000e+00 : f64
  // CHECK-NEXT: return %[[ZERO]] : f64
  %0 = moore.sint_to_real %input : i0 -> f64
  return %0 : !moore.f64
}

// CHECK-LABEL: func.func @ZeroWidthUIntToReal(
func.func @ZeroWidthUIntToReal(%input: !moore.i0) -> !moore.f32 {
  // CHECK: %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
  // CHECK-NEXT: return %[[ZERO]] : f32
  %0 = moore.uint_to_real %input : i0 -> f32
  return %0 : !moore.f32
}
