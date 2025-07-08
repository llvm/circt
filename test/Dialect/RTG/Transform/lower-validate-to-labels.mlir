// RUN: circt-opt --rtg-lower-validate-to-labels --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: rtg.test @basic
rtg.test @basic() {
  // CHECK-NEXT: [[REG:%.+]] = rtg.fixed_reg #rtgtest.t1
  // CHECK-NEXT: [[IMM:%.+]] = rtg.constant #rtg.isa.immediate<32, 0>
  // CHECK-NEXT: [[LBL:%.+]] = rtg.label_decl "spike.pre.printreg.x5.id1"
  // CHECK-NEXT: rtg.label global [[LBL]]
  // CHECK-NEXT: rtgtest.rv32i.lui [[REG]], [[IMM]] : !rtg.isa.immediate<32>
  %0 = rtg.fixed_reg #rtgtest.t0
  %1 = rtg.fixed_reg #rtgtest.t1
  %2 = rtg.constant #rtg.isa.immediate<32, 0>
  %3 = rtg.validate %0, %2, "id1" : !rtgtest.ireg -> !rtg.isa.immediate<32>
  rtgtest.rv32i.lui %1, %3 : !rtg.isa.immediate<32>
}

// -----

rtg.test @no_id() {
  %0 = rtg.fixed_reg #rtgtest.t0
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
