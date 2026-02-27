// RUN: circt-opt --pass-pipeline="builtin.module(rtg.test(rtg-linear-scan-register-allocation))" --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @test0
rtg.test @test0() {
  rtg.isa.segment text {
    // CHECK-DAG: [[V0:%.+]] = rtg.constant #rtgtest.s0
    // CHECK-DAG: [[V1:%.+]] = rtg.constant #rtgtest.s1
    // CHECK-DAG: [[V2:%.+]] = rtg.constant #rtgtest.ra
    // CHECK-DAG: [[V3:%.+]] = rtg.constant #rtgtest.s0
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
}

// CHECK-LABEL: @withFixedRegs
rtg.test @withFixedRegs() {
  rtg.isa.segment text {
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
}

// CHECK-LABEL: @validation
rtg.test @validation() {
  rtg.isa.segment text {
    %reg = rtg.virtual_reg [#rtgtest.ra, #rtgtest.s0, #rtgtest.s1]
    %default = rtg.constant #rtg.isa.immediate<32, 0>
    // CHECK: rtg.validate
    %0 = rtg.validate %reg, %default, "some_id" : !rtgtest.ireg -> !rtg.isa.immediate<32>
    rtgtest.rv32i.lui %reg, %0 : !rtg.isa.immediate<32>
  }
}

// -----

rtg.test @spilling() {
  rtg.isa.segment text {
    // expected-note @below {{cannot choose 'ra' because of overlapping live-range with this register}}
    %0 = rtg.virtual_reg [#rtgtest.ra]
    %imm = rtg.constant #rtg.isa.immediate<12, 0>
    // expected-error @below {{no register available for allocation within constraints}}
    %1 = rtg.virtual_reg [#rtgtest.ra]
    // expected-note @below {{live range starts here}}
    // expected-note @below {{live range ends here}}
    rtgtest.rv32i.jalr %0, %1, %imm
  }
}

// -----

// CHECK-LABEL: @nonOverlappingRanges
rtg.test @nonOverlappingRanges() {
  rtg.isa.segment text {
    // CHECK: [[V0:%.+]] = rtg.constant #rtgtest.ra
    // CHECK: [[V1:%.+]] = rtg.constant #rtgtest.ra
    // CHECK: rtgtest.rv32i.jalr [[V0]]
    // CHECK: rtgtest.rv32i.jalr [[V1]]
    %0 = rtg.virtual_reg [#rtgtest.ra, #rtgtest.s0, #rtgtest.s1]
    %1 = rtg.virtual_reg [#rtgtest.ra, #rtgtest.s0, #rtgtest.s1]
    %imm = rtg.constant #rtg.isa.immediate<12, 0>
    rtgtest.rv32i.jalr %0, %0, %imm
    rtgtest.rv32i.jalr %1, %1, %imm
  }
}

// CHECK-LABEL: @registerConstraintsMultipleDependents
rtg.test @registerConstraintsMultipleDependents() {
  rtg.isa.segment text {
    // CHECK: [[V0:%.+]] = rtg.constant #rtgtest.a2
    // CHECK: [[V1:%.+]] = rtg.isa.register_to_index [[V0]]
    // CHECK: [[V2:%.+]] = index.constant 1
    // CHECK: [[V5:%.+]] = index.sub [[V1]], [[V2]]
    // CHECK: [[V6:%.+]] = index.add [[V1]], [[V2]]
    // CHECK: [[V7:%.+]] = rtg.isa.index_to_register [[V5]]
    // CHECK: [[V8:%.+]] = rtg.isa.index_to_register [[V6]]
    // CHECK: [[V3:%.+]] = rtg.constant #rtgtest.a0
    // CHECK: [[V4:%.+]] = rtg.constant #rtgtest.a1
    // CHECK: rtgtest.rv32i.add [[V3]], [[V4]], [[V4]]
    // CHECK: rtgtest.rv32i.add [[V0]], [[V7]], [[V8]]
    // CHECK: rtgtest.rv32i.add [[V3]], [[V3]], [[V3]]
    %0 = rtg.virtual_reg [#rtgtest.a0, #rtgtest.a1, #rtgtest.a2]
    %idx = rtg.isa.register_to_index %0 : !rtgtest.ireg
    %c1 = index.constant 1
    %idx1 = index.sub %idx, %c1
    %idx2 = index.add %idx, %c1
    %1 = rtg.isa.index_to_register %idx1 : !rtgtest.ireg
    %2 = rtg.isa.index_to_register %idx2 : !rtgtest.ireg
    %3 = rtg.virtual_reg [#rtgtest.a0, #rtgtest.a1, #rtgtest.a2]
    %4 = rtg.virtual_reg [#rtgtest.a0, #rtgtest.a1, #rtgtest.a2]
    rtgtest.rv32i.add %3, %4, %4
    rtgtest.rv32i.add %0, %1, %2
    rtgtest.rv32i.add %3, %3, %3
  }
}

// CHECK-LABEL: @multipleSegments
rtg.test @multipleSegments() {
  rtg.isa.segment data {
    // Data segments are not visited
    %data_reg = rtg.virtual_reg [#rtgtest.ra, #rtgtest.s0]
    rtgtest.rv32i.add %data_reg, %data_reg, %data_reg
  }
  rtg.isa.segment text {
    // CHECK: [[V0:%.+]] = rtg.constant #rtgtest.ra
    %0 = rtg.virtual_reg [#rtgtest.ra, #rtgtest.s0]
    // CHECK: [[V1:%.+]] = rtg.constant #rtgtest.s0
    %1 = rtg.virtual_reg [#rtgtest.s0, #rtgtest.s1]
    // CHECK: rtgtest.rv32i.add [[V0]], [[V0]], [[V1]]
    rtgtest.rv32i.add %0, %0, %1
  }
  rtg.isa.segment text {
    // CHECK: [[V0:%.+]] = rtg.constant #rtgtest.ra
    %0 = rtg.virtual_reg [#rtgtest.ra, #rtgtest.s0]
    // CHECK: [[V1:%.+]] = rtg.constant #rtgtest.s0
    %1 = rtg.virtual_reg [#rtgtest.s0, #rtgtest.s1]
    // CHECK: rtgtest.rv32i.add [[V0]], [[V0]], [[V1]]
    rtgtest.rv32i.add %0, %0, %1
  }
}

// CHECK-LABEL: @noUsers
rtg.test @noUsers() {
  rtg.isa.segment text {
    // This virtual register has no users and should be removed
    %unused = rtg.virtual_reg [#rtgtest.ra]
    // CHECK: [[V0:%.+]] = rtg.constant #rtgtest.ra
    %0 = rtg.virtual_reg [#rtgtest.ra]
    // CHECK: rtgtest.rv32i.add [[V0]], [[V0]], [[V0]]
    rtgtest.rv32i.add %0, %0, %0
  }
}

// CHECK-LABEL: @constraintOpSuccess
rtg.test @constraintOpSuccess() {
  rtg.isa.segment text {
    // CHECK: [[V0:%.+]] = rtg.constant #rtgtest.s2
    %0 = rtg.virtual_reg [#rtgtest.s1, #rtgtest.s2]
    %idx = rtg.isa.register_to_index %0 : !rtgtest.ireg
    %c0 = index.constant 0
    %c2 = index.constant 2
    %mod2 = index.remu %idx, %c2
    %even = index.cmp eq(%mod2, %c0)
    rtg.constraint %even
    rtgtest.rv32i.add %0, %0, %0
  }
}

// -----

rtg.test @registerConstraintsMultipleDependents() {
  rtg.isa.segment text {
    // expected-note @below {{cannot choose 'a0' because of overlapping live-range with this register}}
    %0 = rtg.virtual_reg [#rtgtest.a0, #rtgtest.a1, #rtgtest.a2]
    %idx = rtg.isa.register_to_index %0 : !rtgtest.ireg
    %c1 = index.constant 1
    %c2 = index.constant 2
    %idx1 = index.add %idx, %c1
    %idx2 = index.add %idx, %c2
    // expected-note @below {{cannot choose 'a1' because of overlapping live-range with this register}}
    %1 = rtg.isa.index_to_register %idx1 : !rtgtest.ireg
    // expected-note @below {{cannot choose 'a2' because of overlapping live-range with this register}}
    %2 = rtg.isa.index_to_register %idx2 : !rtgtest.ireg
    // expected-error @below {{no register available for allocation within constraints}}
    %3 = rtg.virtual_reg [#rtgtest.a0, #rtgtest.a1, #rtgtest.a2]
    %4 = rtg.virtual_reg [#rtgtest.a0, #rtgtest.a1, #rtgtest.a2]
    rtgtest.rv32i.add %0, %1, %2
    // expected-note @below {{live range starts here}}
    // expected-note @below {{live range ends here}}
    rtgtest.rv32i.add %0, %4, %3
    rtgtest.rv32i.add %0, %1, %2
  }
}

// -----

rtg.test @constraintViolation() {
  rtg.isa.segment text {
    // expected-error @below {{no register available for allocation within constraints}}
    %0 = rtg.virtual_reg [#rtgtest.ra, #rtgtest.s0]
    %idx = rtg.isa.register_to_index %0 : !rtgtest.ireg
    %c100 = index.constant 100
    %is_impossible = index.cmp eq(%idx, %c100)
    // Constraint that the register index must be 100 (impossible - max is 31)
    // expected-note @below {{constraint would be violated when choosing 'ra'}}
    // expected-note @below {{constraint would be violated when choosing 's0'}}
    rtg.constraint %is_impossible
    // expected-note @below {{live range starts here}}
    // expected-note @below {{live range ends here}}
    rtgtest.rv32i.add %0, %0, %0
  }
}
