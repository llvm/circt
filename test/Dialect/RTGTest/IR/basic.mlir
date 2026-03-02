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

  // CHECK: rtg.virtual_reg [#rtgtest.ra : !rtgtest.ireg, #rtgtest.sp : !rtgtest.ireg]
  rtg.virtual_reg [#rtgtest.ra, #rtgtest.sp]
}

// CHECK-LABEL: @instructions
// CHECK-SAME: (imm = [[IMM:%.+]]: !rtg.isa.immediate<32>, mem = [[MEM:%.+]]: !rtg.isa.memory<32>, rd = [[RD:%.+]]: !rtgtest.ireg, rs = [[RS:%.+]]: !rtgtest.ireg)
rtg.test @instructions(imm = %imm: !rtg.isa.immediate<32>, mem = %mem: !rtg.isa.memory<32>, rd = %rd: !rtgtest.ireg, rs = %rs: !rtgtest.ireg) {
  // CHECK: rtgtest.two_register_instr [[RD]], [[RS]]
  rtgtest.two_register_instr %rd, %rs
  // CHECK: rtgtest.three_register_instr [[RD]], [[RS]], [[RS]]
  rtgtest.three_register_instr %rd, %rs, %rs
  // CHECK: rtgtest.zero_operand_instr
  rtgtest.zero_operand_instr
  // CHECK: rtgtest.immediate_instr [[RD]], [[IMM]]
  rtgtest.immediate_instr %rd, %imm
  // CHECK: rtgtest.memory_instr [[RD]], [[MEM]] : !rtg.isa.memory<32>
  rtgtest.memory_instr %rd, %mem : !rtg.isa.memory<32>
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
