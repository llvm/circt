// RUN: circt-opt --pass-pipeline="builtin.module(rtg.test(rtg-linear-scan-register-allocation))" --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @test0
rtg.test @test0() {
  // CHECK: [[V0:%.+]] = rtg.constant #rtgtest.ra
  // CHECK: [[V1:%.+]] = rtg.constant #rtgtest.s1
  // CHECK: [[V2:%.+]] = rtg.constant #rtgtest.s0
  // CHECK: [[V3:%.+]] = rtg.constant #rtgtest.ra
  // CHECK: rtgtest.rv32i.jalr [[V0]], [[V2]]
  // CHECK: rtgtest.rv32i.jalr [[V1]], [[V0]]
  // CHECK: rtgtest.rv32i.jalr [[V3]], [[V1]]
  // CHECK: rtgtest.rv32i.jalr [[V2]], [[V3]]
  %0 = rtg.virtual_reg [#rtgtest.ra, #rtgtest.s0, #rtgtest.s1]
  %1 = rtg.virtual_reg [#rtgtest.ra, #rtgtest.s0, #rtgtest.s1]
  %2 = rtg.virtual_reg [#rtgtest.ra, #rtgtest.s0, #rtgtest.s1]
  %3 = rtg.virtual_reg [#rtgtest.ra, #rtgtest.s0, #rtgtest.s1]
  %imm = rtg.constant #rtg.isa.immediate<12, 0>
  rtgtest.rv32i.jalr %0, %2, %imm
  rtgtest.rv32i.jalr %1, %0, %imm
  rtgtest.rv32i.jalr %3, %1, %imm
  rtgtest.rv32i.jalr %2, %3, %imm
}

// CHECK-LABEL: @withFixedRegs
rtg.test @withFixedRegs() {
  // CHECK: [[V0:%.+]] = rtg.constant #rtgtest.ra
  // CHECK: [[V1:%.+]] = rtg.constant #rtgtest.s1
  // CHECK: [[V2:%.+]] = rtg.constant #rtgtest.s0
  // CHECK: [[V3:%.+]] = rtg.constant #rtgtest.ra
  // CHECK: rtgtest.rv32i.jalr [[V0]], [[V2]]
  // CHECK: rtgtest.rv32i.jalr [[V1]], [[V0]]
  // CHECK: rtgtest.rv32i.jalr [[V3]], [[V1]]
  // CHECK: rtgtest.rv32i.jalr [[V2]], [[V3]]
  %0 = rtg.constant #rtgtest.ra
  %1 = rtg.virtual_reg [#rtgtest.ra, #rtgtest.s0, #rtgtest.s1]
  %2 = rtg.constant #rtgtest.s0
  %3 = rtg.virtual_reg [#rtgtest.ra, #rtgtest.s0, #rtgtest.s1]
  %imm = rtg.constant #rtg.isa.immediate<12, 0>
  rtgtest.rv32i.jalr %0, %2, %imm
  rtgtest.rv32i.jalr %1, %0, %imm
  rtgtest.rv32i.jalr %3, %1, %imm
  rtgtest.rv32i.jalr %2, %3, %imm
}

// CHECK-LABEL: @validation
rtg.test @validation() {
  %reg = rtg.virtual_reg [#rtgtest.ra, #rtgtest.s0, #rtgtest.s1]
  %default = rtg.constant #rtg.isa.immediate<32, 0>
  // CHECK: rtg.validate
  %0 = rtg.validate %reg, %default, "some_id" : !rtgtest.ireg -> !rtg.isa.immediate<32>
  rtgtest.rv32i.lui %reg, %0 : !rtg.isa.immediate<32>
}

// -----

rtg.test @spilling() {
  %0 = rtg.virtual_reg [#rtgtest.ra]
  // expected-error @below {{need to spill this register, but not supported yet}}
  %1 = rtg.virtual_reg [#rtgtest.ra]
  %imm = rtg.constant #rtg.isa.immediate<12, 0>
  rtgtest.rv32i.jalr %0, %1, %imm
}

// -----

rtg.test @unsupportedUser() {
  %0 = rtg.virtual_reg [#rtgtest.ra]
  // expected-error @below {{only operations implementing 'RegisterAllocationOpInterface' are allowed to use registers}}
  rtg.set_create %0 : !rtgtest.ireg
}
