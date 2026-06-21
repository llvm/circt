// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

// CHECK-LABEL: func.func @zero_width_sext
// CHECK-SAME: (%arg0: i0) -> i1
func.func @zero_width_sext(%z: !moore.i0) -> !moore.i1 {
  // CHECK: %[[ZERO:.*]] = hw.constant false
  %0 = moore.sext %z : i0 -> i1
  // CHECK: return %[[ZERO]] : i1
  return %0 : !moore.i1
}
