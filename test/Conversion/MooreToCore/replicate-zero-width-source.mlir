// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

// CHECK-LABEL: func.func @zero_width_replicate_source
// CHECK-SAME: (%arg0: i0) -> i4
func.func @zero_width_replicate_source(%zero: !moore.i0) -> !moore.i4 {
  // CHECK-NEXT: %[[ZERO:.*]] = hw.constant 0 : i4
  %0 = moore.replicate %zero : !moore.i0 -> !moore.i4
  // CHECK-NEXT: return %[[ZERO]] : i4
  return %0 : !moore.i4
}
