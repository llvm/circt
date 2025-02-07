// RUN: circt-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// CHECK-LABEL: @cpus
// CHECK-SAME: !rtgtest.cpu
rtg.target @cpus : !rtg.dict<cpu: !rtgtest.cpu> {
  // CHECK: rtgtest.cpu_decl <0>
  %0 = rtgtest.cpu_decl <0>
  rtg.yield %0 : !rtgtest.cpu
}

rtg.test @misc : !rtg.dict<> {
  // CHECK: rtgtest.constant_test i32 {value = "str"}
  %0 = rtgtest.constant_test i32 {value = "str"}
}

// CHECK-LABEL: rtg.test @registers
// CHECK-SAME: !rtgtest.ireg
rtg.test @registers : !rtg.dict<reg: !rtgtest.ireg> {
^bb0(%reg: !rtgtest.ireg):
  // CHECK: rtg.fixed_reg #rtgtest.zero : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.ra : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.sp : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.gp : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.tp : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.t0 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.t1 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.t2 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.s0 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.s1 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.a0 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.a1 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.a2 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.a3 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.a4 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.a5 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.a6 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.a7 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.s2 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.s3 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.s4 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.s5 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.s6 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.s7 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.s8 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.s9 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.s10 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.s11 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.t3 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.t4 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.t5 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.t6 : !rtgtest.ireg
  rtg.fixed_reg #rtgtest.zero
  rtg.fixed_reg #rtgtest.ra
  rtg.fixed_reg #rtgtest.sp
  rtg.fixed_reg #rtgtest.gp
  rtg.fixed_reg #rtgtest.tp
  rtg.fixed_reg #rtgtest.t0
  rtg.fixed_reg #rtgtest.t1
  rtg.fixed_reg #rtgtest.t2
  rtg.fixed_reg #rtgtest.s0
  rtg.fixed_reg #rtgtest.s1
  rtg.fixed_reg #rtgtest.a0
  rtg.fixed_reg #rtgtest.a1
  rtg.fixed_reg #rtgtest.a2
  rtg.fixed_reg #rtgtest.a3
  rtg.fixed_reg #rtgtest.a4
  rtg.fixed_reg #rtgtest.a5
  rtg.fixed_reg #rtgtest.a6
  rtg.fixed_reg #rtgtest.a7
  rtg.fixed_reg #rtgtest.s2
  rtg.fixed_reg #rtgtest.s3
  rtg.fixed_reg #rtgtest.s4
  rtg.fixed_reg #rtgtest.s5
  rtg.fixed_reg #rtgtest.s6
  rtg.fixed_reg #rtgtest.s7
  rtg.fixed_reg #rtgtest.s8
  rtg.fixed_reg #rtgtest.s9
  rtg.fixed_reg #rtgtest.s10
  rtg.fixed_reg #rtgtest.s11
  rtg.fixed_reg #rtgtest.t3
  rtg.fixed_reg #rtgtest.t4
  rtg.fixed_reg #rtgtest.t5
  rtg.fixed_reg #rtgtest.t6

  // CHECK: rtg.virtual_reg [#rtgtest.ra : !rtgtest.ireg, #rtgtest.sp : !rtgtest.ireg]
  rtg.virtual_reg [#rtgtest.ra, #rtgtest.sp]
}

// CHECK-LABEL: @immediates
rtg.test @immediates : !rtg.dict<> {
  // CHECK: rtgtest.immediate #rtgtest.imm5<3> : !rtgtest.imm5
  rtgtest.immediate #rtgtest.imm5<3> : !rtgtest.imm5
  // CHECK: rtgtest.immediate #rtgtest.imm12<3> : !rtgtest.imm12
  rtgtest.immediate #rtgtest.imm12<3> : !rtgtest.imm12
  // CHECK: rtgtest.immediate #rtgtest.imm13<3> : !rtgtest.imm13
  rtgtest.immediate #rtgtest.imm13<3> : !rtgtest.imm13
  // CHECK: rtgtest.immediate #rtgtest.imm21<3> : !rtgtest.imm21
  rtgtest.immediate #rtgtest.imm21<3> : !rtgtest.imm21
  // CHECK: rtgtest.immediate #rtgtest.imm32<3> : !rtgtest.imm32
  rtgtest.immediate #rtgtest.imm32<3> : !rtgtest.imm32
}

// CHECK-LABEL: @instructions
rtg.test @instructions : !rtg.dict<imm: !rtgtest.imm12, imm13: !rtgtest.imm13, imm21: !rtgtest.imm21, imm32: !rtgtest.imm32, imm5: !rtgtest.imm5, label: !rtg.label, rd: !rtgtest.ireg, rs: !rtgtest.ireg> {
// CHECK: ([[IMM:%.+]]: !rtgtest.imm12, [[IMM13:%.+]]: !rtgtest.imm13, [[IMM21:%.+]]: !rtgtest.imm21, [[IMM32:%.+]]: !rtgtest.imm32, [[IMM5:%.+]]: !rtgtest.imm5, [[LABEL:%.+]]: !rtg.label, [[RD:%.+]]: !rtgtest.ireg, [[RS:%.+]]: !rtgtest.ireg)
^bb0(%imm: !rtgtest.imm12, %imm13: !rtgtest.imm13, %imm21: !rtgtest.imm21, %imm32: !rtgtest.imm32, %imm5: !rtgtest.imm5, %label: !rtg.label, %rd: !rtgtest.ireg, %rs: !rtgtest.ireg):
  // CHECK: rtgtest.rv32i.jalr [[RD]], [[RS]], [[IMM]]
  rtgtest.rv32i.jalr %rd, %rs, %imm
  // CHECK: rtgtest.rv32i.lb [[RD]], [[RS]], [[IMM]]
  rtgtest.rv32i.lb %rd, %rs, %imm
  // CHECK: rtgtest.rv32i.lh [[RD]], [[RS]], [[IMM]]
  rtgtest.rv32i.lh %rd, %rs, %imm
  // CHECK: rtgtest.rv32i.lw [[RD]], [[RS]], [[IMM]]
  rtgtest.rv32i.lw %rd, %rs, %imm
  // CHECK: rtgtest.rv32i.lbu [[RD]], [[RS]], [[IMM]]
  rtgtest.rv32i.lbu %rd, %rs, %imm
  // CHECK: rtgtest.rv32i.lhu [[RD]], [[RS]], [[IMM]]
  rtgtest.rv32i.lhu %rd, %rs, %imm
  // CHECK: rtgtest.rv32i.ecall
  rtgtest.rv32i.ecall
  // CHECK: rtgtest.rv32i.ebreak
  rtgtest.rv32i.ebreak
  // CHECK: rtgtest.rv32i.beq [[RD]], [[RS]], [[IMM13]] : !rtgtest.imm13
  rtgtest.rv32i.beq %rd, %rs, %imm13 : !rtgtest.imm13
  // CHECK: rtgtest.rv32i.bne [[RD]], [[RS]], [[IMM13]] : !rtgtest.imm13
  rtgtest.rv32i.bne %rd, %rs, %imm13 : !rtgtest.imm13
  // CHECK: rtgtest.rv32i.blt [[RD]], [[RS]], [[IMM13]] : !rtgtest.imm13
  rtgtest.rv32i.blt %rd, %rs, %imm13 : !rtgtest.imm13
  // CHECK: rtgtest.rv32i.bge [[RD]], [[RS]], [[IMM13]] : !rtgtest.imm13
  rtgtest.rv32i.bge %rd, %rs, %imm13 : !rtgtest.imm13
  // CHECK: rtgtest.rv32i.bltu [[RD]], [[RS]], [[IMM13]] : !rtgtest.imm13
  rtgtest.rv32i.bltu %rd, %rs, %imm13 : !rtgtest.imm13
  // CHECK: rtgtest.rv32i.bgeu [[RD]], [[RS]], [[IMM13]] : !rtgtest.imm13
  rtgtest.rv32i.bgeu %rd, %rs, %imm13 : !rtgtest.imm13
  // CHECK: rtgtest.rv32i.beq [[RD]], [[RS]], [[LABEL]] : !rtg.label
  rtgtest.rv32i.beq %rd, %rs, %label : !rtg.label
  // CHECK: rtgtest.rv32i.bne [[RD]], [[RS]], [[LABEL]] : !rtg.label
  rtgtest.rv32i.bne %rd, %rs, %label : !rtg.label
  // CHECK: rtgtest.rv32i.blt [[RD]], [[RS]], [[LABEL]] : !rtg.label
  rtgtest.rv32i.blt %rd, %rs, %label : !rtg.label
  // CHECK: rtgtest.rv32i.bge [[RD]], [[RS]], [[LABEL]] : !rtg.label
  rtgtest.rv32i.bge %rd, %rs, %label : !rtg.label
  // CHECK: rtgtest.rv32i.bltu [[RD]], [[RS]], [[LABEL]] : !rtg.label
  rtgtest.rv32i.bltu %rd, %rs, %label : !rtg.label
  // CHECK: rtgtest.rv32i.bgeu [[RD]], [[RS]], [[LABEL]] : !rtg.label
  rtgtest.rv32i.bgeu %rd, %rs, %label : !rtg.label

  // CHECK: rtgtest.rv32i.add [[RD]], [[RS]], [[RS]] {rtg.some_attr}
  rtgtest.rv32i.add %rd, %rs, %rs {rtg.some_attr}
  // CHECK: rtgtest.rv32i.sub [[RD]], [[RS]], [[RS]] {rtg.some_attr}
  rtgtest.rv32i.sub %rd, %rs, %rs {rtg.some_attr}
  // CHECK: rtgtest.rv32i.sll [[RD]], [[RS]], [[RS]] {rtg.some_attr}
  rtgtest.rv32i.sll %rd, %rs, %rs {rtg.some_attr}
  // CHECK: rtgtest.rv32i.slt [[RD]], [[RS]], [[RS]] {rtg.some_attr}
  rtgtest.rv32i.slt %rd, %rs, %rs {rtg.some_attr}
  // CHECK: rtgtest.rv32i.sltu [[RD]], [[RS]], [[RS]] {rtg.some_attr}
  rtgtest.rv32i.sltu %rd, %rs, %rs {rtg.some_attr}
  // CHECK: rtgtest.rv32i.xor [[RD]], [[RS]], [[RS]] {rtg.some_attr}
  rtgtest.rv32i.xor %rd, %rs, %rs {rtg.some_attr}
  // CHECK: rtgtest.rv32i.srl [[RD]], [[RS]], [[RS]] {rtg.some_attr}
  rtgtest.rv32i.srl %rd, %rs, %rs {rtg.some_attr}
  // CHECK: rtgtest.rv32i.sra [[RD]], [[RS]], [[RS]] {rtg.some_attr}
  rtgtest.rv32i.sra %rd, %rs, %rs {rtg.some_attr}
  // CHECK: rtgtest.rv32i.or [[RD]], [[RS]], [[RS]] {rtg.some_attr}
  rtgtest.rv32i.or %rd, %rs, %rs {rtg.some_attr}
  // CHECK: rtgtest.rv32i.and [[RD]], [[RS]], [[RS]] {rtg.some_attr}
  rtgtest.rv32i.and %rd, %rs, %rs {rtg.some_attr}

  // CHECK: rtgtest.rv32i.sb [[RD]], [[RS]], [[IMM]] {rtg.some_attr}
  rtgtest.rv32i.sb %rd, %rs, %imm {rtg.some_attr}
  // CHECK: rtgtest.rv32i.sh [[RD]], [[RS]], [[IMM]] {rtg.some_attr}
  rtgtest.rv32i.sh %rd, %rs, %imm {rtg.some_attr}
  // CHECK: rtgtest.rv32i.sw [[RD]], [[RS]], [[IMM]] {rtg.some_attr}
  rtgtest.rv32i.sw %rd, %rs, %imm {rtg.some_attr}

  // CHECK: rtgtest.rv32i.lui [[RD]], [[IMM32]] : !rtgtest.imm32 {rtg.some_attr}
  rtgtest.rv32i.lui %rd, %imm32 : !rtgtest.imm32 {rtg.some_attr}
  // CHECK: rtgtest.rv32i.auipc [[RD]], [[IMM32]] : !rtgtest.imm32 {rtg.some_attr}
  rtgtest.rv32i.auipc %rd, %imm32 : !rtgtest.imm32 {rtg.some_attr}
  // CHECK: rtgtest.rv32i.jal [[RD]], [[IMM21]] : !rtgtest.imm21 {rtg.some_attr}
  rtgtest.rv32i.jal %rd, %imm21 : !rtgtest.imm21 {rtg.some_attr}

  // CHECK: rtgtest.rv32i.lui [[RD]], [[LABEL]] : !rtg.label {rtg.some_attr}
  rtgtest.rv32i.lui %rd, %label : !rtg.label {rtg.some_attr}
  // CHECK: rtgtest.rv32i.auipc [[RD]], [[LABEL]] : !rtg.label {rtg.some_attr}
  rtgtest.rv32i.auipc %rd, %label : !rtg.label {rtg.some_attr}
  // CHECK: rtgtest.rv32i.jal [[RD]], [[LABEL]] : !rtg.label {rtg.some_attr}
  rtgtest.rv32i.jal %rd, %label : !rtg.label {rtg.some_attr}

  // CHECK: rtgtest.rv32i.addi [[RD]], [[RS]], [[IMM]] {rtg.some_attr}
  rtgtest.rv32i.addi %rd, %rs, %imm {rtg.some_attr}
  // CHECK: rtgtest.rv32i.slti [[RD]], [[RS]], [[IMM]] {rtg.some_attr}
  rtgtest.rv32i.slti %rd, %rs, %imm {rtg.some_attr}
  // CHECK: rtgtest.rv32i.sltiu [[RD]], [[RS]], [[IMM]] {rtg.some_attr}
  rtgtest.rv32i.sltiu %rd, %rs, %imm {rtg.some_attr}
  // CHECK: rtgtest.rv32i.xori [[RD]], [[RS]], [[IMM]] {rtg.some_attr}
  rtgtest.rv32i.xori %rd, %rs, %imm {rtg.some_attr}
  // CHECK: rtgtest.rv32i.ori [[RD]], [[RS]], [[IMM]] {rtg.some_attr}
  rtgtest.rv32i.ori %rd, %rs, %imm {rtg.some_attr}
  // CHECK: rtgtest.rv32i.andi [[RD]], [[RS]], [[IMM]] {rtg.some_attr}
  rtgtest.rv32i.andi %rd, %rs, %imm {rtg.some_attr}

  // CHECK: rtgtest.rv32i.slli [[RD]], [[RS]], [[IMM5]] {rtg.some_attr}
  rtgtest.rv32i.slli %rd, %rs, %imm5 {rtg.some_attr}
  // CHECK: rtgtest.rv32i.srli [[RD]], [[RS]], [[IMM5]] {rtg.some_attr}
  rtgtest.rv32i.srli %rd, %rs, %imm5 {rtg.some_attr}
  // CHECK: rtgtest.rv32i.srai [[RD]], [[RS]], [[IMM5]] {rtg.some_attr}
  rtgtest.rv32i.srai %rd, %rs, %imm5 {rtg.some_attr}
}

// -----

rtg.test @immediateTooBig : !rtg.dict<> {
  // expected-error @below {{cannot represent 2000000 with 12 bits}}
  rtgtest.immediate #rtgtest.imm12<2000000> : !rtgtest.imm12
}

// -----

rtg.test @emptyAllowed : !rtg.dict<> {
  // expected-error @below {{must have at least one allowed register}}
  rtg.virtual_reg []
}

// -----

rtg.test @invalidAllowedAttr : !rtg.dict<> {
  // expected-error @below {{allowed register attributes must be of RegisterAttrInterface}}
  rtg.virtual_reg ["invalid"]
}
