// RUN: circt-opt --rtg-lower-validate-to-labels --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: rtg.test @basic
rtg.test @basic() {
  // CHECK-NEXT: [[REG:%.+]] = rtg.constant #rtgtest.t1
  // CHECK-NEXT: [[IMM0:%.+]] = rtg.constant #rtg.isa.immediate<32, 0>
  // CHECK-NEXT: [[LBL:%.+]] = rtg.constant #rtg.isa.label<"spike.pre.printreg.x5.id1">
  // CHECK-NEXT: rtg.label global [[LBL]]
  // CHECK-NEXT: rtgtest.rv32i.lui [[REG]], [[IMM0]] : !rtg.isa.immediate<32>
  %0 = rtg.constant #rtgtest.t0
  %1 = rtg.constant #rtgtest.t1
  %2 = rtg.constant #rtg.isa.immediate<32, 0>
  %3 = rtg.validate %0, %2, "id1" : !rtgtest.ireg -> !rtg.isa.immediate<32>
  rtgtest.rv32i.lui %1, %3 : !rtg.isa.immediate<32>

  // CHECK-NEXT: [[IMM1:%.+]] = rtg.constant #rtg.isa.immediate<32, 1>
  // CHECK-NEXT: [[IMM2:%.+]] = rtg.constant #rtg.isa.immediate<32, 2>
  // CHECK-NEXT: [[LBL:%.+]] = rtg.constant #rtg.isa.label<"spike.pre.printreg.x5.id2">
  // CHECK-NEXT: rtg.label global [[LBL]]
  // CHECK-NEXT: rtgtest.rv32i.lui [[REG]], [[IMM0]] : !rtg.isa.immediate<32>
  // CHECK-NEXT: rtgtest.rv32i.lui [[REG]], [[IMM1]] : !rtg.isa.immediate<32>
  // CHECK-NEXT: rtgtest.rv32i.lui [[REG]], [[IMM2]] : !rtg.isa.immediate<32>
  %4 = rtg.constant #rtg.isa.immediate<32, 1>
  %5 = rtg.constant #rtg.isa.immediate<32, 2>
  %6 = rtg.constant #rtg.isa.immediate<32, 3>
  %7 = rtg.constant #rtg.isa.immediate<32, 4>
  %8:3 = rtg.validate %0, %2, "id2" (%4, %5 else %6, %7 : !rtg.isa.immediate<32>, !rtg.isa.immediate<32>) : !rtgtest.ireg -> !rtg.isa.immediate<32>
  rtgtest.rv32i.lui %1, %8#0 : !rtg.isa.immediate<32>
  rtgtest.rv32i.lui %1, %8#1 : !rtg.isa.immediate<32>
  rtgtest.rv32i.lui %1, %8#2 : !rtg.isa.immediate<32>
}

// -----

rtg.test @no_id() {
  %0 = rtg.constant #rtgtest.t0
  %1 = rtg.constant #rtg.isa.immediate<32, 0>
  // expected-error @below {{expected ID to be set}}
  %2 = rtg.validate %0, %1 : !rtgtest.ireg -> !rtg.isa.immediate<32>
}

// -----

rtg.test @no_reg() {
  %0 = rtg.virtual_reg [#rtgtest.t0, #rtgtest.t1]
  %1 = rtg.constant #rtg.isa.immediate<32, 0>
  // expected-error @below {{could not determine register}}
  %2 = rtg.validate %0, %1 : !rtgtest.ireg -> !rtg.isa.immediate<32>
}
