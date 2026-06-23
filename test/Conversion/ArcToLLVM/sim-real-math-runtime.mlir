// RUN: circt-opt %s --lower-arc-to-llvm | FileCheck %s

// CHECK-DAG: llvm.func @arcRuntimeIR_realHypot(f64, f64) -> f64
func.func @real_hypot(%lhs: f64, %rhs: f64) -> f64 {
  // CHECK: llvm.call @arcRuntimeIR_realHypot(%{{.*}}, %{{.*}}) : (f64, f64) -> f64
  %result = sim.real.hypot %lhs, %rhs : f64
  return %result : f64
}
