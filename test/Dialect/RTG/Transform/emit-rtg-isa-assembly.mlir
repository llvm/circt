// RUN: circt-opt --rtg-emit-isa-assembly %s 2>&1 >/dev/null | FileCheck %s --check-prefix=CHECK-ALLOWED --match-full-lines --strict-whitespace
// RUN: circt-opt --rtg-emit-isa-assembly="unsupported-instructions=rtgtest.rv32i.ebreak,rtgtest.rv32i.ecall unsupported-instructions-file=%S/unsupported-instr.txt" %s 2>&1 >/dev/null | FileCheck %s --match-full-lines --strict-whitespace

// CHECK:# Begin of test0
// CHECK-EMPTY:
// CHECK-ALLOWED:# Begin of test0
// CHECK-ALLOWED-EMPTY:

rtg.test @test0 : !rtg.dict<> {
  %rd = rtg.fixed_reg #rtgtest.ra
  %rs = rtg.fixed_reg #rtgtest.s0
  %imm = rtgtest.immediate #rtgtest.imm12<0>
  %imm13 = rtgtest.immediate #rtgtest.imm13<6144>

  // CHECK-ALLOWED-NEXT:    jalr ra, 0(s0)
  // CHECK-NEXT:    # jalr ra, 0(s0)
  // CHECK-NEXT:    .word 0x400E7
  rtgtest.rv32i.jalr %rd, %rs, %imm

  // CHECK-ALLOWED-NEXT:    lb ra, 0(s0)
  // CHECK-NEXT:    # lb ra, 0(s0)
  // CHECK-NEXT:    .word 0x40083
  rtgtest.rv32i.lb %rd, %rs, %imm

  // CHECK-ALLOWED-NEXT:    lh ra, 0(s0)
  // CHECK-NEXT:    # lh ra, 0(s0)
  // CHECK-NEXT:    .word 0x41083
  rtgtest.rv32i.lh %rd, %rs, %imm

  // CHECK-ALLOWED-NEXT:    lw ra, 0(s0)
  // CHECK-NEXT:    # lw ra, 0(s0)
  // CHECK-NEXT:    .word 0x42083
  rtgtest.rv32i.lw %rd, %rs, %imm

  // CHECK-ALLOWED-NEXT:    lbu ra, 0(s0)
  // CHECK-NEXT:    # lbu ra, 0(s0)
  // CHECK-NEXT:    .word 0x44083
  rtgtest.rv32i.lbu %rd, %rs, %imm

  // CHECK-ALLOWED-NEXT:    lhu ra, 0(s0)
  // CHECK-NEXT:    # lhu ra, 0(s0)
  // CHECK-NEXT:    .word 0x45083
  rtgtest.rv32i.lhu %rd, %rs, %imm

  // CHECK-ALLOWED-NEXT:    ebreak
  // CHECK-NEXT:    # ebreak
  // CHECK-NEXT:    .word 0x100073
  rtgtest.rv32i.ebreak

  // CHECK-ALLOWED-NEXT:    ecall
  // CHECK-NEXT:    # ecall
  // CHECK-NEXT:    .word 0x73
  rtgtest.rv32i.ecall

  // CHECK-ALLOWED-NEXT:    beq ra, s0, 6144
  // CHECK-NEXT:    # beq ra, s0, 6144
  // CHECK-NEXT:    .word 0x808080E3
  rtgtest.rv32i.beq %rd, %rs, %imm13 : !rtgtest.imm13

  // CHECK-ALLOWED-NEXT:    bne ra, s0, 6144
  // CHECK-NEXT:    # bne ra, s0, 6144
  // CHECK-NEXT:    .word 0x808090E3
  rtgtest.rv32i.bne %rd, %rs, %imm13 : !rtgtest.imm13

  // CHECK-ALLOWED-NEXT:    blt ra, s0, 6144
  // CHECK-NEXT:    # blt ra, s0, 6144
  // CHECK-NEXT:    .word 0x8080C0E3
  rtgtest.rv32i.blt %rd, %rs, %imm13 : !rtgtest.imm13

  // CHECK-ALLOWED-NEXT:    bge ra, s0, 6144
  // CHECK-NEXT:    # bge ra, s0, 6144
  // CHECK-NEXT:    .word 0x8080D0E3
  rtgtest.rv32i.bge %rd, %rs, %imm13 : !rtgtest.imm13

  // CHECK-ALLOWED-NEXT:    bltu ra, s0, 6144
  // CHECK-NEXT:    # bltu ra, s0, 6144
  // CHECK-NEXT:    .word 0x8080E0E3
  rtgtest.rv32i.bltu %rd, %rs, %imm13 : !rtgtest.imm13

  // CHECK-ALLOWED-NEXT:    bgeu ra, s0, 6144
  // CHECK-NEXT:    # bgeu ra, s0, 6144
  // CHECK-NEXT:    .word 0x8080F0E3
  rtgtest.rv32i.bgeu %rd, %rs, %imm13 : !rtgtest.imm13

  // CHECK-ALLOWED-NEXT:    add ra, s0, s0
  // CHECK-NEXT:    # add ra, s0, s0
  // CHECK-NEXT:    .word 0x8400B3
  rtgtest.rv32i.add %rd, %rs, %rs

  // CHECK-ALLOWED-NEXT:    sub ra, s0, s0
  // CHECK-NEXT:    # sub ra, s0, s0
  // CHECK-NEXT:    .word 0x408400B3
  rtgtest.rv32i.sub %rd, %rs, %rs

  // CHECK-ALLOWED-NEXT:    sll ra, s0, s0
  // CHECK-NEXT:    # sll ra, s0, s0
  // CHECK-NEXT:    .word 0x8410B3
  rtgtest.rv32i.sll %rd, %rs, %rs

  // CHECK-ALLOWED-NEXT:    slt ra, s0, s0
  // CHECK-NEXT:    # slt ra, s0, s0
  // CHECK-NEXT:    .word 0x8420B3
  rtgtest.rv32i.slt %rd, %rs, %rs

  // CHECK-ALLOWED-NEXT:    sltu ra, s0, s0
  // CHECK-NEXT:    # sltu ra, s0, s0
  // CHECK-NEXT:    .word 0x8430B3
  rtgtest.rv32i.sltu %rd, %rs, %rs

  // CHECK-ALLOWED-NEXT:    xor ra, s0, s0
  // CHECK-NEXT:    # xor ra, s0, s0
  // CHECK-NEXT:    .word 0x8440B3
  rtgtest.rv32i.xor %rd, %rs, %rs

  // CHECK-ALLOWED-NEXT:    srl ra, s0, s0
  // CHECK-NEXT:    # srl ra, s0, s0
  // CHECK-NEXT:    .word 0x8450B3
  rtgtest.rv32i.srl %rd, %rs, %rs

  // CHECK-ALLOWED-NEXT:    sra ra, s0, s0
  // CHECK-NEXT:    # sra ra, s0, s0
  // CHECK-NEXT:    .word 0x408450B3
  rtgtest.rv32i.sra %rd, %rs, %rs

  // CHECK-ALLOWED-NEXT:    or ra, s0, s0
  // CHECK-NEXT:    # or ra, s0, s0
  // CHECK-NEXT:    .word 0x8460B3
  rtgtest.rv32i.or %rd, %rs, %rs

  // CHECK-ALLOWED-NEXT:    and ra, s0, s0
  // CHECK-NEXT:    # and ra, s0, s0
  // CHECK-NEXT:    .word 0x8470B3
  rtgtest.rv32i.and %rd, %rs, %rs
}

// CHECK-EMPTY:
// CHECK-NEXT:# End of test0
// CHECK-ALLOWED-EMPTY:
// CHECK-ALLOWED-NEXT:# End of test0
