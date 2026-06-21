// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

// CHECK-LABEL: func.func @BoolCastString
// CHECK-SAME: (%arg0: !sim.dstring) -> i1
func.func @BoolCastString(%arg0: !moore.string) -> !moore.i1 {
  // CHECK: [[LENGTH:%.+]] = sim.string.length %arg0
  // CHECK: [[ZERO:%.+]] = hw.constant 0 : i64
  // CHECK: [[BOOL:%.+]] = comb.icmp ne [[LENGTH]], [[ZERO]] : i64
  %0 = moore.bool_cast %arg0 : !moore.string -> !moore.i1
  // CHECK: return [[BOOL]] : i1
  return %0 : !moore.i1
}
