// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

// CHECK-LABEL: func.func @BoolCastTime
// CHECK-SAME: (%arg0: !llhd.time) -> i1
func.func @BoolCastTime(%arg0: !moore.time) -> !moore.l1 {
  // CHECK: [[INT:%.+]] = llhd.time_to_int %arg0
  // CHECK: [[ZERO:%.+]] = hw.constant 0 : i64
  // CHECK: [[CMP:%.+]] = comb.icmp ne [[INT]], [[ZERO]] : i64
  // CHECK: return [[CMP]] : i1
  %0 = moore.bool_cast %arg0 : !moore.time -> !moore.l1
  return %0 : !moore.l1
}
