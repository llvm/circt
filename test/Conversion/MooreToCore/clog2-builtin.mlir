// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

module {
  // CHECK-LABEL: func.func @clog2_builtin(
  // CHECK-SAME: %[[INPUT:.+]]: i8
  func.func @clog2_builtin(%input: !moore.i8) -> !moore.i8 {
    // CHECK: %[[ZERO:.+]] = hw.constant 0 : i8
    // CHECK: %[[GT:.+]] = comb.icmp ugt %[[INPUT]],
    // CHECK: %[[ONE:.+]] = hw.constant 1 : i8
    // CHECK: comb.mux %[[GT]], %[[ONE]], %[[ZERO]] : i8
    // CHECK: hw.constant 8 : i8
    // CHECK-NOT: moore.builtin.clog2
    %result = moore.builtin.clog2 %input : i8
    return %result : !moore.i8
  }
}
