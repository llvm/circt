// RUN: circt-opt --rtg-lower-validate-to-labels --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: rtg.test @basic
rtg.test @basic() {
  // CHECK-NEXT: [[REG:%.+]] = rtg.constant #rtgtest.t1
  // CHECK-NEXT: [[IMM0:%.+]] = rtg.constant 0 : i32
  // CHECK-NEXT: [[LBL:%.+]] = rtg.constant #rtg.isa.label<"spike.pre.printreg.x5.id1">
  // CHECK-NEXT: rtg.label global [[LBL]]
  // CHECK-NEXT: rtgtest.lui [[REG]], [[IMM0]] : i32
  %0 = rtg.constant #rtgtest.t0
  %1 = rtg.constant #rtgtest.t1
  %2 = rtg.constant 0 : i32
  %3 = rtg.validate %0, %2, "id1" : !rtgtest.ireg -> i32
  rtgtest.lui %1, %3 : i32

  // CHECK-NEXT: [[IMM1:%.+]] = rtg.constant 1 : i32
  // CHECK-NEXT: [[IMM2:%.+]] = rtg.constant 2 : i32
  // CHECK-NEXT: [[LBL:%.+]] = rtg.constant #rtg.isa.label<"spike.pre.printreg.x5.id2">
  // CHECK-NEXT: rtg.label global [[LBL]]
  // CHECK-NEXT: rtgtest.lui [[REG]], [[IMM0]] : i32
  // CHECK-NEXT: rtgtest.lui [[REG]], [[IMM1]] : i32
  // CHECK-NEXT: rtgtest.lui [[REG]], [[IMM2]] : i32
  %4 = rtg.constant 1 : i32
  %5 = rtg.constant 2 : i32
  %6 = rtg.constant 3 : i32
  %7 = rtg.constant 4 : i32
  %8:3 = rtg.validate %0, %2, "id2" (%4, %5 else %6, %7 : i32, i32) : !rtgtest.ireg -> i32
  rtgtest.lui %1, %8#0 : i32
  rtgtest.lui %1, %8#1 : i32
  rtgtest.lui %1, %8#2 : i32
}

// -----

rtg.test @no_id() {
  %0 = rtg.constant #rtgtest.t0
  %1 = rtg.constant 0 : i32
  // expected-error @below {{expected ID to be set}}
  %2 = rtg.validate %0, %1 : !rtgtest.ireg -> i32
}

// -----

rtg.test @no_reg() {
  %0 = rtg.virtual_reg [#rtgtest.t0, #rtgtest.t1]
  %1 = rtg.constant 0 : i32
  // expected-error @below {{could not determine register}}
  %2 = rtg.validate %0, %1 : !rtgtest.ireg -> i32
}
