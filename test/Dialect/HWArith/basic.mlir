// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

// CHECK-LABEL: @test1() {
hw.module @test1() {
  // CHECK: %0 = hwarith.constant 0 : ui1
  // CHECK: %1 = hwarith.constant 1 : ui1
  // CHECK: %2 = hwarith.constant 2 : ui2
  // CHECK: %3 = hwarith.constant 22 : ui5
  %0 = hwarith.constant 0 : ui1
  %1 = hwarith.constant 1 : ui1
  %2 = hwarith.constant 2 : ui2
  %3 = hwarith.constant 22 : ui5

  // CHECK: %4 = hwarith.add %1, %1 : (ui1, ui1) -> ui2
  %4 = hwarith.add %1, %1 : (ui1, ui1) -> ui2
  // CHECK: %5 = hwarith.add %0, %2 : (ui1, ui2) -> ui3
  %5 = hwarith.add %0, %2 : (ui1, ui2) -> ui3
  // CHECK: %6 = hwarith.mul %2, %3 : (ui2, ui5) -> ui7
  %6 = hwarith.mul %2, %3 : (ui2, ui5) -> ui7
  // CHECK: %7 = hwarith.add %6, %1 : (ui7, ui1) -> ui8
  %7 = hwarith.add %6, %1 : (ui7, ui1) -> ui8
  // CHECK: %8 = hwarith.sub %7, %2 : (ui8, ui2) -> si9
  %8 = hwarith.sub %7, %2 : (ui8, ui2) -> si9
  // CHECK: %9 = hwarith.div %8, %1 : (si9, ui1) -> si9
  %9 = hwarith.div %8, %1 : (si9, ui1) -> si9
  // CHECK: %10 = hwarith.add %9, %5 : (si9, ui3) -> si10
  %10 = hwarith.add %9, %5 : (si9, ui3) -> si10
}
