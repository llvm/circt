// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

// CHECK-LABEL: @addSameSignSameSize(%clk: i1, %rst: i1) {
hw.module @addSameSignSameSize(%clk: i1, %rst: i1) {
  // CHECK:  %0 = apint.constant 0 : ui1
  // CHECK:  %1 = apint.constant 1 : ui1
  // CHECK:  %2 = apint.add %1, %1 : ui1, ui1
  // CHECK:  %3 = apint.add %0, %2 : ui1, ui2
  %0 = apint.constant 0 : ui1
  %1 = apint.constant 1 : ui1
  %2 = apint.add %1, %1 : ui1, ui1
  %3 = apint.add %0, %2 : ui1, ui2
}

// CHECK-LABEL: @mulSameSignSameSize(%clk: i1, %rst: i1) {
hw.module @mulSameSignSameSize(%clk: i1, %rst: i1) {
  // CHECK:  %0 = apint.constant 2 : ui2
  // CHECK:  %1 = apint.constant 22 : ui5
  // CHECK:  %2 = apint.constant 1 : ui1
  // CHECK:  %3 = apint.mul %0, %1 : ui2, ui5
  // CHECK:  %4 = apint.add %3, %2 : ui7, ui1
  %0 = apint.constant 2 : ui2
  %1 = apint.constant 22 : ui5
  %2 = apint.constant 1 : ui1
  %3 = apint.mul %0, %1 : ui2, ui5
  %4 = apint.add %3, %2 : ui7, ui1
}

