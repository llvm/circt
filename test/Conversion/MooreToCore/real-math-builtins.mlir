// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

// CHECK-LABEL: func.func @RealMathBuiltins
// CHECK-SAME: (%arg0: f64) -> f64
func.func @RealMathBuiltins(%value: !moore.f64) -> !moore.f64 {
  // CHECK: %[[LN:.*]] = math.log %arg0 : f64
  %ln = moore.builtin.ln %value : !moore.f64
  // CHECK: %[[LOG10:.*]] = math.log10 %[[LN]] : f64
  %log10 = moore.builtin.log10 %ln : !moore.f64
  // CHECK: %[[EXP:.*]] = math.exp %[[LOG10]] : f64
  %exp = moore.builtin.exp %log10 : !moore.f64
  // CHECK: %[[SQRT:.*]] = math.sqrt %[[EXP]] : f64
  %sqrt = moore.builtin.sqrt %exp : !moore.f64
  // CHECK: %[[FLOOR:.*]] = math.floor %[[SQRT]] : f64
  %floor = moore.builtin.floor %sqrt : !moore.f64
  // CHECK: %[[CEIL:.*]] = math.ceil %[[FLOOR]] : f64
  %ceil = moore.builtin.ceil %floor : !moore.f64
  // CHECK: %[[SIN:.*]] = math.sin %[[CEIL]] : f64
  %sin = moore.builtin.sin %ceil : !moore.f64
  // CHECK: %[[COS:.*]] = math.cos %[[SIN]] : f64
  %cos = moore.builtin.cos %sin : !moore.f64
  // CHECK: %[[TAN:.*]] = math.tan %[[COS]] : f64
  %tan = moore.builtin.tan %cos : !moore.f64
  // CHECK: %[[ASIN:.*]] = math.asin %[[TAN]] : f64
  %asin = moore.builtin.asin %tan : !moore.f64
  // CHECK: %[[ACOS:.*]] = math.acos %[[ASIN]] : f64
  %acos = moore.builtin.acos %asin : !moore.f64
  // CHECK: %[[ATAN:.*]] = math.atan %[[ACOS]] : f64
  %atan = moore.builtin.atan %acos : !moore.f64
  // CHECK: %[[ATAN2:.*]] = math.atan2 %[[ATAN]], %[[ATAN]] : f64
  %atan2 = moore.builtin.atan2 %atan, %atan : !moore.f64
  // CHECK: %[[SINH:.*]] = math.sinh %[[ATAN2]] : f64
  %sinh = moore.builtin.sinh %atan2 : !moore.f64
  // CHECK: %[[COSH:.*]] = math.cosh %[[SINH]] : f64
  %cosh = moore.builtin.cosh %sinh : !moore.f64
  // CHECK: %[[TANH:.*]] = math.tanh %[[COSH]] : f64
  %tanh = moore.builtin.tanh %cosh : !moore.f64
  // CHECK: %[[ASINH:.*]] = math.asinh %[[TANH]] : f64
  %asinh = moore.builtin.asinh %tanh : !moore.f64
  // CHECK: %[[ACOSH:.*]] = math.acosh %[[ASINH]] : f64
  %acosh = moore.builtin.acosh %asinh : !moore.f64
  // CHECK: %[[ATANH:.*]] = math.atanh %[[ACOSH]] : f64
  %atanh = moore.builtin.atanh %acosh : !moore.f64
  // CHECK: %[[HYPOT:.*]] = sim.real.hypot %[[ATANH]], %[[ATANH]] : f64
  %hypot = moore.builtin.hypot %atanh, %atanh : !moore.f64
  // CHECK: return %[[HYPOT]] : f64
  return %hypot : !moore.f64
}
