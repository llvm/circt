// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

// CHECK-LABEL: func.func @zero_width_replicate
// CHECK-SAME: (%arg0: i1) -> i0
func.func @zero_width_replicate(%one: !moore.i1) -> !moore.i0 {
  // CHECK: %[[ZERO:.*]] = hw.constant 0 : i0
  %0 = moore.replicate %one : i1 -> i0
  // CHECK: return %[[ZERO]] : i0
  return %0 : !moore.i0
}
