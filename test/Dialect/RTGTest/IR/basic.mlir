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
  // CHECK: rtg.constant #rtgtest.zero : !rtgtest.ireg
  // CHECK: rtg.constant #rtgtest.ra : !rtgtest.ireg
  // CHECK: rtg.constant #rtgtest.sp : !rtgtest.ireg
  // CHECK: rtg.constant #rtgtest.gp : !rtgtest.ireg
  // CHECK: rtg.constant #rtgtest.tp : !rtgtest.ireg
  // CHECK: rtg.constant #rtgtest.t0 : !rtgtest.ireg
  // CHECK: rtg.constant #rtgtest.t1 : !rtgtest.ireg
  // CHECK: rtg.constant #rtgtest.t2 : !rtgtest.ireg
  // CHECK: rtg.constant #rtgtest.s0 : !rtgtest.ireg
  // CHECK: rtg.constant #rtgtest.s1 : !rtgtest.ireg
  // CHECK: rtg.constant #rtgtest.a0 : !rtgtest.ireg
  // CHECK: rtg.constant #rtgtest.a1 : !rtgtest.ireg
  // CHECK: rtg.constant #rtgtest.a2 : !rtgtest.ireg
  // CHECK: rtg.constant #rtgtest.a3 : !rtgtest.ireg
  // CHECK: rtg.constant #rtgtest.a4 : !rtgtest.ireg
  // CHECK: rtg.constant #rtgtest.a5 : !rtgtest.ireg
  // CHECK: rtg.constant #rtgtest.a6 : !rtgtest.ireg
  // CHECK: rtg.constant #rtgtest.a7 : !rtgtest.ireg
  // CHECK: rtg.constant #rtgtest.s2 : !rtgtest.ireg
  // CHECK: rtg.constant #rtgtest.s3 : !rtgtest.ireg
  // CHECK: rtg.constant #rtgtest.s4 : !rtgtest.ireg
  // CHECK: rtg.constant #rtgtest.s5 : !rtgtest.ireg
  // CHECK: rtg.constant #rtgtest.s6 : !rtgtest.ireg
  // CHECK: rtg.constant #rtgtest.s7 : !rtgtest.ireg
  // CHECK: rtg.constant #rtgtest.s8 : !rtgtest.ireg
  // CHECK: rtg.constant #rtgtest.s9 : !rtgtest.ireg
  // CHECK: rtg.constant #rtgtest.s10 : !rtgtest.ireg
  // CHECK: rtg.constant #rtgtest.s11 : !rtgtest.ireg
  // CHECK: rtg.constant #rtgtest.t3 : !rtgtest.ireg
  // CHECK: rtg.constant #rtgtest.t4 : !rtgtest.ireg
  // CHECK: rtg.constant #rtgtest.t5 : !rtgtest.ireg
  // CHECK: rtg.constant #rtgtest.t6 : !rtgtest.ireg
  rtg.constant #rtgtest.zero
  rtg.constant #rtgtest.ra
  rtg.constant #rtgtest.sp
  rtg.constant #rtgtest.gp
  rtg.constant #rtgtest.tp
  rtg.constant #rtgtest.t0
  rtg.constant #rtgtest.t1
  rtg.constant #rtgtest.t2
  rtg.constant #rtgtest.s0
  rtg.constant #rtgtest.s1
  rtg.constant #rtgtest.a0
  rtg.constant #rtgtest.a1
  rtg.constant #rtgtest.a2
  rtg.constant #rtgtest.a3
  rtg.constant #rtgtest.a4
  rtg.constant #rtgtest.a5
  rtg.constant #rtgtest.a6
  rtg.constant #rtgtest.a7
  rtg.constant #rtgtest.s2
  rtg.constant #rtgtest.s3
  rtg.constant #rtgtest.s4
  rtg.constant #rtgtest.s5
  rtg.constant #rtgtest.s6
  rtg.constant #rtgtest.s7
  rtg.constant #rtgtest.s8
  rtg.constant #rtgtest.s9
  rtg.constant #rtgtest.s10
  rtg.constant #rtgtest.s11
  rtg.constant #rtgtest.t3
  rtg.constant #rtgtest.t4
  rtg.constant #rtgtest.t5
  rtg.constant #rtgtest.t6

  // CHECK: rtg.virtual_reg [#rtgtest.ra : !rtgtest.ireg, #rtgtest.sp : !rtgtest.ireg]
  rtg.virtual_reg [#rtgtest.ra, #rtgtest.sp]
}

// CHECK-LABEL: @instructions
// CHECK-SAME: (imm = [[IMM:%.+]]: !rtg.isa.immediate<12>, imm13 = [[IMM13:%.+]]: !rtg.isa.immediate<13>, imm21 = [[IMM21:%.+]]: !rtg.isa.immediate<21>, imm32 = [[IMM32:%.+]]: !rtg.isa.immediate<32>, imm5 = [[IMM5:%.+]]: !rtg.isa.immediate<5>, label = [[LABEL:%.+]]: !rtg.isa.label, rd = [[RD:%.+]]: !rtgtest.ireg, rs = [[RS:%.+]]: !rtgtest.ireg)
rtg.test @instructions(imm = %imm: !rtg.isa.immediate<12>, imm13 = %imm13: !rtg.isa.immediate<13>, imm21 = %imm21: !rtg.isa.immediate<21>, imm32 = %imm32: !rtg.isa.immediate<32>, imm5 = %imm5: !rtg.isa.immediate<5>, label = %label: !rtg.isa.label, rd = %rd: !rtgtest.ireg, rs = %rs: !rtgtest.ireg) {
  // CHECK: rtgtest.jalr [[RD]], [[RS]], [[IMM]]
  rtgtest.jalr %rd, %rs, %imm
  // CHECK: rtgtest.lb [[RD]], [[RS]], [[IMM]]
  rtgtest.lb %rd, %rs, %imm
  // CHECK: rtgtest.lh [[RD]], [[RS]], [[IMM]]
  rtgtest.lh %rd, %rs, %imm
  // CHECK: rtgtest.lw [[RD]], [[RS]], [[IMM]]
  rtgtest.lw %rd, %rs, %imm
  // CHECK: rtgtest.lbu [[RD]], [[RS]], [[IMM]]
  rtgtest.lbu %rd, %rs, %imm
  // CHECK: rtgtest.lhu [[RD]], [[RS]], [[IMM]]
  rtgtest.lhu %rd, %rs, %imm
  // CHECK: rtgtest.ecall
  rtgtest.ecall
  // CHECK: rtgtest.ebreak
  rtgtest.ebreak
  // CHECK: rtgtest.beq [[RD]], [[RS]], [[IMM13]] : !rtg.isa.immediate<13>
  rtgtest.beq %rd, %rs, %imm13 : !rtg.isa.immediate<13>
  // CHECK: rtgtest.bne [[RD]], [[RS]], [[IMM13]] : !rtg.isa.immediate<13>
  rtgtest.bne %rd, %rs, %imm13 : !rtg.isa.immediate<13>
  // CHECK: rtgtest.blt [[RD]], [[RS]], [[IMM13]] : !rtg.isa.immediate<13>
  rtgtest.blt %rd, %rs, %imm13 : !rtg.isa.immediate<13>
  // CHECK: rtgtest.bge [[RD]], [[RS]], [[IMM13]] : !rtg.isa.immediate<13>
  rtgtest.bge %rd, %rs, %imm13 : !rtg.isa.immediate<13>
  // CHECK: rtgtest.bltu [[RD]], [[RS]], [[IMM13]] : !rtg.isa.immediate<13>
  rtgtest.bltu %rd, %rs, %imm13 : !rtg.isa.immediate<13>
  // CHECK: rtgtest.bgeu [[RD]], [[RS]], [[IMM13]] : !rtg.isa.immediate<13>
  rtgtest.bgeu %rd, %rs, %imm13 : !rtg.isa.immediate<13>
  // CHECK: rtgtest.beq [[RD]], [[RS]], [[LABEL]] : !rtg.isa.label
  rtgtest.beq %rd, %rs, %label : !rtg.isa.label
  // CHECK: rtgtest.bne [[RD]], [[RS]], [[LABEL]] : !rtg.isa.label
  rtgtest.bne %rd, %rs, %label : !rtg.isa.label
  // CHECK: rtgtest.blt [[RD]], [[RS]], [[LABEL]] : !rtg.isa.label
  rtgtest.blt %rd, %rs, %label : !rtg.isa.label
  // CHECK: rtgtest.bge [[RD]], [[RS]], [[LABEL]] : !rtg.isa.label
  rtgtest.bge %rd, %rs, %label : !rtg.isa.label
  // CHECK: rtgtest.bltu [[RD]], [[RS]], [[LABEL]] : !rtg.isa.label
  rtgtest.bltu %rd, %rs, %label : !rtg.isa.label
  // CHECK: rtgtest.bgeu [[RD]], [[RS]], [[LABEL]] : !rtg.isa.label
  rtgtest.bgeu %rd, %rs, %label : !rtg.isa.label

  // CHECK: rtgtest.add [[RD]], [[RS]], [[RS]] {rtg.some_attr}
  rtgtest.add %rd, %rs, %rs {rtg.some_attr}
  // CHECK: rtgtest.sub [[RD]], [[RS]], [[RS]] {rtg.some_attr}
  rtgtest.sub %rd, %rs, %rs {rtg.some_attr}
  // CHECK: rtgtest.sll [[RD]], [[RS]], [[RS]] {rtg.some_attr}
  rtgtest.sll %rd, %rs, %rs {rtg.some_attr}
  // CHECK: rtgtest.slt [[RD]], [[RS]], [[RS]] {rtg.some_attr}
  rtgtest.slt %rd, %rs, %rs {rtg.some_attr}
  // CHECK: rtgtest.sltu [[RD]], [[RS]], [[RS]] {rtg.some_attr}
  rtgtest.sltu %rd, %rs, %rs {rtg.some_attr}
  // CHECK: rtgtest.xor [[RD]], [[RS]], [[RS]] {rtg.some_attr}
  rtgtest.xor %rd, %rs, %rs {rtg.some_attr}
  // CHECK: rtgtest.srl [[RD]], [[RS]], [[RS]] {rtg.some_attr}
  rtgtest.srl %rd, %rs, %rs {rtg.some_attr}
  // CHECK: rtgtest.sra [[RD]], [[RS]], [[RS]] {rtg.some_attr}
  rtgtest.sra %rd, %rs, %rs {rtg.some_attr}
  // CHECK: rtgtest.or [[RD]], [[RS]], [[RS]] {rtg.some_attr}
  rtgtest.or %rd, %rs, %rs {rtg.some_attr}
  // CHECK: rtgtest.and [[RD]], [[RS]], [[RS]] {rtg.some_attr}
  rtgtest.and %rd, %rs, %rs {rtg.some_attr}

  // CHECK: rtgtest.sb [[RD]], [[RS]], [[IMM]] {rtg.some_attr}
  rtgtest.sb %rd, %rs, %imm {rtg.some_attr}
  // CHECK: rtgtest.sh [[RD]], [[RS]], [[IMM]] {rtg.some_attr}
  rtgtest.sh %rd, %rs, %imm {rtg.some_attr}
  // CHECK: rtgtest.sw [[RD]], [[RS]], [[IMM]] {rtg.some_attr}
  rtgtest.sw %rd, %rs, %imm {rtg.some_attr}

  // CHECK: rtgtest.lui [[RD]], [[IMM32]] : !rtg.isa.immediate<32> {rtg.some_attr}
  rtgtest.lui %rd, %imm32 : !rtg.isa.immediate<32> {rtg.some_attr}
  // CHECK: rtgtest.auipc [[RD]], [[IMM32]] : !rtg.isa.immediate<32> {rtg.some_attr}
  rtgtest.auipc %rd, %imm32 : !rtg.isa.immediate<32> {rtg.some_attr}
  // CHECK: rtgtest.jal [[RD]], [[IMM21]] : !rtg.isa.immediate<21> {rtg.some_attr}
  rtgtest.jal %rd, %imm21 : !rtg.isa.immediate<21> {rtg.some_attr}

  // CHECK: rtgtest.lui [[RD]], [[LABEL]] : !rtg.isa.label {rtg.some_attr}
  rtgtest.lui %rd, %label : !rtg.isa.label {rtg.some_attr}
  // CHECK: rtgtest.auipc [[RD]], [[LABEL]] : !rtg.isa.label {rtg.some_attr}
  rtgtest.auipc %rd, %label : !rtg.isa.label {rtg.some_attr}
  // CHECK: rtgtest.jal [[RD]], [[LABEL]] : !rtg.isa.label {rtg.some_attr}
  rtgtest.jal %rd, %label : !rtg.isa.label {rtg.some_attr}

  // CHECK: rtgtest.addi [[RD]], [[RS]], [[IMM]] {rtg.some_attr}
  rtgtest.addi %rd, %rs, %imm {rtg.some_attr}
  // CHECK: rtgtest.slti [[RD]], [[RS]], [[IMM]] {rtg.some_attr}
  rtgtest.slti %rd, %rs, %imm {rtg.some_attr}
  // CHECK: rtgtest.sltiu [[RD]], [[RS]], [[IMM]] {rtg.some_attr}
  rtgtest.sltiu %rd, %rs, %imm {rtg.some_attr}
  // CHECK: rtgtest.xori [[RD]], [[RS]], [[IMM]] {rtg.some_attr}
  rtgtest.xori %rd, %rs, %imm {rtg.some_attr}
  // CHECK: rtgtest.ori [[RD]], [[RS]], [[IMM]] {rtg.some_attr}
  rtgtest.ori %rd, %rs, %imm {rtg.some_attr}
  // CHECK: rtgtest.andi [[RD]], [[RS]], [[IMM]] {rtg.some_attr}
  rtgtest.andi %rd, %rs, %imm {rtg.some_attr}

  // CHECK: rtgtest.slli [[RD]], [[RS]], [[IMM5]] {rtg.some_attr}
  rtgtest.slli %rd, %rs, %imm5 {rtg.some_attr}
  // CHECK: rtgtest.srli [[RD]], [[RS]], [[IMM5]] {rtg.some_attr}
  rtgtest.srli %rd, %rs, %imm5 {rtg.some_attr}
  // CHECK: rtgtest.srai [[RD]], [[RS]], [[IMM5]] {rtg.some_attr}
  rtgtest.srai %rd, %rs, %imm5 {rtg.some_attr}
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
