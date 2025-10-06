// RUN: circt-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// CHECK-LABEL: @cpus
// CHECK-SAME: !rtgtest.cpu
rtg.target @cpus : !rtg.dict<cpu: !rtgtest.cpu, hartid: index> {
  // CHECK: [[V0:%.+]] = rtg.constant #rtgtest.cpu<0> : !rtgtest.cpu
  %0 = rtg.constant #rtgtest.cpu<0>
  // CHECK: rtgtest.get_hartid [[V0]]
  %1 = rtgtest.get_hartid %0
  rtg.yield %0, %1 : !rtgtest.cpu, index
}

rtg.test @misc() {
  // CHECK: rtgtest.constant_test i32 {value = "str"}
  %0 = rtgtest.constant_test i32 {value = "str"}
}

// CHECK-LABEL: rtg.test @registers
// CHECK-SAME: !rtgtest.ireg
rtg.test @registers(reg = %reg: !rtgtest.ireg) {
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

// CHECK-LABEL: @instructions
// CHECK-SAME: (imm = [[IMM:%.+]]: !rtg.isa.immediate<12>, imm13 = [[IMM13:%.+]]: !rtg.isa.immediate<13>, imm21 = [[IMM21:%.+]]: !rtg.isa.immediate<21>, imm32 = [[IMM32:%.+]]: !rtg.isa.immediate<32>, imm5 = [[IMM5:%.+]]: !rtg.isa.immediate<5>, label = [[LABEL:%.+]]: !rtg.isa.label, rd = [[RD:%.+]]: !rtgtest.ireg, rs = [[RS:%.+]]: !rtgtest.ireg)
rtg.test @instructions(imm = %imm: !rtg.isa.immediate<12>, imm13 = %imm13: !rtg.isa.immediate<13>, imm21 = %imm21: !rtg.isa.immediate<21>, imm32 = %imm32: !rtg.isa.immediate<32>, imm5 = %imm5: !rtg.isa.immediate<5>, label = %label: !rtg.isa.label, rd = %rd: !rtgtest.ireg, rs = %rs: !rtgtest.ireg) {
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
  // CHECK: rtgtest.rv32i.beq [[RD]], [[RS]], [[IMM13]] : !rtg.isa.immediate<13>
  rtgtest.rv32i.beq %rd, %rs, %imm13 : !rtg.isa.immediate<13>
  // CHECK: rtgtest.rv32i.bne [[RD]], [[RS]], [[IMM13]] : !rtg.isa.immediate<13>
  rtgtest.rv32i.bne %rd, %rs, %imm13 : !rtg.isa.immediate<13>
  // CHECK: rtgtest.rv32i.blt [[RD]], [[RS]], [[IMM13]] : !rtg.isa.immediate<13>
  rtgtest.rv32i.blt %rd, %rs, %imm13 : !rtg.isa.immediate<13>
  // CHECK: rtgtest.rv32i.bge [[RD]], [[RS]], [[IMM13]] : !rtg.isa.immediate<13>
  rtgtest.rv32i.bge %rd, %rs, %imm13 : !rtg.isa.immediate<13>
  // CHECK: rtgtest.rv32i.bltu [[RD]], [[RS]], [[IMM13]] : !rtg.isa.immediate<13>
  rtgtest.rv32i.bltu %rd, %rs, %imm13 : !rtg.isa.immediate<13>
  // CHECK: rtgtest.rv32i.bgeu [[RD]], [[RS]], [[IMM13]] : !rtg.isa.immediate<13>
  rtgtest.rv32i.bgeu %rd, %rs, %imm13 : !rtg.isa.immediate<13>
  // CHECK: rtgtest.rv32i.beq [[RD]], [[RS]], [[LABEL]] : !rtg.isa.label
  rtgtest.rv32i.beq %rd, %rs, %label : !rtg.isa.label
  // CHECK: rtgtest.rv32i.bne [[RD]], [[RS]], [[LABEL]] : !rtg.isa.label
  rtgtest.rv32i.bne %rd, %rs, %label : !rtg.isa.label
  // CHECK: rtgtest.rv32i.blt [[RD]], [[RS]], [[LABEL]] : !rtg.isa.label
  rtgtest.rv32i.blt %rd, %rs, %label : !rtg.isa.label
  // CHECK: rtgtest.rv32i.bge [[RD]], [[RS]], [[LABEL]] : !rtg.isa.label
  rtgtest.rv32i.bge %rd, %rs, %label : !rtg.isa.label
  // CHECK: rtgtest.rv32i.bltu [[RD]], [[RS]], [[LABEL]] : !rtg.isa.label
  rtgtest.rv32i.bltu %rd, %rs, %label : !rtg.isa.label
  // CHECK: rtgtest.rv32i.bgeu [[RD]], [[RS]], [[LABEL]] : !rtg.isa.label
  rtgtest.rv32i.bgeu %rd, %rs, %label : !rtg.isa.label

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

  // CHECK: rtgtest.rv32i.lui [[RD]], [[IMM32]] : !rtg.isa.immediate<32> {rtg.some_attr}
  rtgtest.rv32i.lui %rd, %imm32 : !rtg.isa.immediate<32> {rtg.some_attr}
  // CHECK: rtgtest.rv32i.auipc [[RD]], [[IMM32]] : !rtg.isa.immediate<32> {rtg.some_attr}
  rtgtest.rv32i.auipc %rd, %imm32 : !rtg.isa.immediate<32> {rtg.some_attr}
  // CHECK: rtgtest.rv32i.jal [[RD]], [[IMM21]] : !rtg.isa.immediate<21> {rtg.some_attr}
  rtgtest.rv32i.jal %rd, %imm21 : !rtg.isa.immediate<21> {rtg.some_attr}

  // CHECK: rtgtest.rv32i.lui [[RD]], [[LABEL]] : !rtg.isa.label {rtg.some_attr}
  rtgtest.rv32i.lui %rd, %label : !rtg.isa.label {rtg.some_attr}
  // CHECK: rtgtest.rv32i.auipc [[RD]], [[LABEL]] : !rtg.isa.label {rtg.some_attr}
  rtgtest.rv32i.auipc %rd, %label : !rtg.isa.label {rtg.some_attr}
  // CHECK: rtgtest.rv32i.jal [[RD]], [[LABEL]] : !rtg.isa.label {rtg.some_attr}
  rtgtest.rv32i.jal %rd, %label : !rtg.isa.label {rtg.some_attr}

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

rtg.test @emptyAllowed() {
  // expected-error @below {{expected attribute value}}
  // expected-error @below {{failed to parse VirtualRegisterConfigAttr parameter 'allowedRegs' which is to be a `::llvm::ArrayRef<rtg::RegisterAttrInterface>`}}
  rtg.virtual_reg #rtg.virtual_register_config[]
}

// -----

rtg.test @invalidAllowedAttr() {
  // expected-error @below {{invalid kind of attribute specified}}
  // expected-error @below {{failed to parse VirtualRegisterConfigAttr parameter 'allowedRegs' which is to be a `::llvm::ArrayRef<rtg::RegisterAttrInterface>`}}
  rtg.virtual_reg #rtg.virtual_register_config["invalid"]
}

// -----

rtg.test @invalidAllowedAttr() {
  // expected-error @below {{all allowed registers must be of the same type}}
  rtg.virtual_reg #rtg.virtual_register_config[#rtgtest.a0, #rtgtest.f0]
}
