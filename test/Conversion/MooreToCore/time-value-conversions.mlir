// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

// CHECK-LABEL: func.func @time_to_i64
// CHECK-SAME: (%arg0: !llhd.time) -> i64
func.func @time_to_i64(%input: !moore.time) -> !moore.l64 {
  // CHECK: llhd.time_to_int %arg0
  %result = moore.conversion %input : !moore.time -> !moore.l64
  return %result : !moore.l64
}

// CHECK-LABEL: func.func @i64_to_time
// CHECK-SAME: (%arg0: i64) -> !llhd.time
func.func @i64_to_time(%input: !moore.l64) -> !moore.time {
  // CHECK: llhd.int_to_time %arg0
  %result = moore.conversion %input : !moore.l64 -> !moore.time
  return %result : !moore.time
}
