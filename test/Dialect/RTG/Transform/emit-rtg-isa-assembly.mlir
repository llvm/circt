// RUN: circt-opt --rtg-emit-isa-assembly %s 2>&1 >/dev/null | FileCheck %s --check-prefix=CHECK-ALLOWED --match-full-lines --strict-whitespace
// RUN: circt-opt --rtg-emit-isa-assembly="unsupported-instructions=rtgtest.rv32i.ebreak,rtgtest.rv32i.ecall unsupported-instructions-file=%S/unsupported-instr.txt" %s 2>&1 >/dev/null | FileCheck %s --match-full-lines --strict-whitespace

// CHECK:    # Begin of test0
// CHECK-ALLOWED:    # Begin of test0

emit.file "" {
  rtg.comment "Begin of test0"

  %idx8 = index.constant 8
  // CHECK-ALLOWED-NEXT:    .space 8
  // CHECK-NEXT:    .space 8
  rtg.isa.space %idx8

  // CHECK-ALLOWED-NEXT:    .asciz "hello world\n\t\\\""
  // CHECK-NEXT:    .asciz "hello world\n\t\\\""
  rtg.isa.string_data "hello world\n\t\\\""

  %rd = rtg.constant #rtgtest.ra
  %rs = rtg.constant #rtgtest.s0
  %imm = rtg.constant #rtg.isa.immediate<12, 0>
  %imm5 = rtg.constant #rtg.isa.immediate<5, 31>
  %imm21 = rtg.constant #rtg.isa.immediate<21, 0>
  %imm32 = rtg.constant #rtg.isa.immediate<32, 0>
  %neg_imm = rtg.constant #rtg.isa.immediate<12, 4080>
  %imm13 = rtg.constant #rtg.isa.immediate<13, 6144>

  // CHECK-ALLOWED-NEXT:    la ra, 0
  // CHECK-NEXT:    la ra, 0
  rtgtest.rv32i.la %rd, %imm32 : !rtg.isa.immediate<32>

  // CHECK-ALLOWED-NEXT:    jalr ra, -16(s0)
  // CHECK-NEXT:    # jalr ra, -16(s0)
  // CHECK-NEXT:    .word 0xFF0400E7
  rtgtest.rv32i.jalr %rd, %rs, %neg_imm

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
  rtgtest.rv32i.beq %rd, %rs, %imm13 : !rtg.isa.immediate<13>

  // CHECK-ALLOWED-NEXT:    bne ra, s0, 6144
  // CHECK-NEXT:    # bne ra, s0, 6144
  // CHECK-NEXT:    .word 0x808090E3
  rtgtest.rv32i.bne %rd, %rs, %imm13 : !rtg.isa.immediate<13>

  // CHECK-ALLOWED-NEXT:    blt ra, s0, 6144
  // CHECK-NEXT:    # blt ra, s0, 6144
  // CHECK-NEXT:    .word 0x8080C0E3
  rtgtest.rv32i.blt %rd, %rs, %imm13 : !rtg.isa.immediate<13>

  // CHECK-ALLOWED-NEXT:    bge ra, s0, 6144
  // CHECK-NEXT:    # bge ra, s0, 6144
  // CHECK-NEXT:    .word 0x8080D0E3
  rtgtest.rv32i.bge %rd, %rs, %imm13 : !rtg.isa.immediate<13>

  // CHECK-ALLOWED-NEXT:    bltu ra, s0, 6144
  // CHECK-NEXT:    # bltu ra, s0, 6144
  // CHECK-NEXT:    .word 0x8080E0E3
  rtgtest.rv32i.bltu %rd, %rs, %imm13 : !rtg.isa.immediate<13>

  // CHECK-ALLOWED-NEXT:    bgeu ra, s0, 6144
  // CHECK-NEXT:    # bgeu ra, s0, 6144
  // CHECK-NEXT:    .word 0x8080F0E3
  rtgtest.rv32i.bgeu %rd, %rs, %imm13 : !rtg.isa.immediate<13>

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

  // CHECK-ALLOWED-NEXT:    sb ra, -16(s0)
  // CHECK-NEXT:    # sb ra, -16(s0)
  // CHECK-NEXT:    .word 0xFE808823
  rtgtest.rv32i.sb %rd, %rs, %neg_imm

  // CHECK-ALLOWED-NEXT:    sh ra, 0(s0)
  // CHECK-NEXT:    # sh ra, 0(s0)
  // CHECK-NEXT:    .word 0x809023
  rtgtest.rv32i.sh %rd, %rs, %imm

  // CHECK-ALLOWED-NEXT:    sw ra, 0(s0)
  // CHECK-NEXT:    # sw ra, 0(s0)
  // CHECK-NEXT:    .word 0x80A023
  rtgtest.rv32i.sw %rd, %rs, %imm

  // CHECK-ALLOWED-NEXT:    lui ra, 0
  // CHECK-NEXT:    # lui ra, 0
  // CHECK-NEXT:    .word 0xB7
  rtgtest.rv32i.lui %rd, %imm32 : !rtg.isa.immediate<32>

  // CHECK-ALLOWED-NEXT:    auipc ra, 0
  // CHECK-NEXT:    # auipc ra, 0
  // CHECK-NEXT:    .word 0x97
  rtgtest.rv32i.auipc %rd, %imm32 : !rtg.isa.immediate<32>

  // CHECK-ALLOWED-NEXT:    jal ra, 0
  // CHECK-NEXT:    # jal ra, 0
  // CHECK-NEXT:    .word 0xEF
  rtgtest.rv32i.jal %rd, %imm21 : !rtg.isa.immediate<21>

  // CHECK-ALLOWED-NEXT:    addi ra, s0, -16
  // CHECK-NEXT:    # addi ra, s0, -16
  // CHECK-NEXT:    .word 0xFF040093
  rtgtest.rv32i.addi %rd, %rs, %neg_imm

  // CHECK-ALLOWED-NEXT:    slti ra, s0, 0
  // CHECK-NEXT:    # slti ra, s0, 0
  // CHECK-NEXT:    .word 0x42093
  rtgtest.rv32i.slti %rd, %rs, %imm

  // CHECK-ALLOWED-NEXT:    sltiu ra, s0, 0
  // CHECK-NEXT:    # sltiu ra, s0, 0
  // CHECK-NEXT:    .word 0x43093
  rtgtest.rv32i.sltiu %rd, %rs, %imm

  // CHECK-ALLOWED-NEXT:    xori ra, s0, 0
  // CHECK-NEXT:    # xori ra, s0, 0
  // CHECK-NEXT:    .word 0x44093
  rtgtest.rv32i.xori %rd, %rs, %imm

  // CHECK-ALLOWED-NEXT:    ori ra, s0, 0
  // CHECK-NEXT:    # ori ra, s0, 0
  // CHECK-NEXT:    .word 0x46093
  rtgtest.rv32i.ori %rd, %rs, %imm

  // CHECK-ALLOWED-NEXT:    andi ra, s0, 0
  // CHECK-NEXT:    # andi ra, s0, 0
  // CHECK-NEXT:    .word 0x47093
  rtgtest.rv32i.andi %rd, %rs, %imm

  // CHECK-ALLOWED-NEXT:    slli ra, s0, 31
  // CHECK-NEXT:    # slli ra, s0, 31
  // CHECK-NEXT:    .word 0x1F41093
  rtgtest.rv32i.slli %rd, %rs, %imm5

  // CHECK-ALLOWED-NEXT:    srli ra, s0, 31
  // CHECK-NEXT:    # srli ra, s0, 31
  // CHECK-NEXT:    .word 0x1F45093
  rtgtest.rv32i.srli %rd, %rs, %imm5

  // CHECK-ALLOWED-NEXT:    srai ra, s0, 31
  // CHECK-NEXT:    # srai ra, s0, 31
  // CHECK-NEXT:    .word 0x41F45093
  rtgtest.rv32i.srai %rd, %rs, %imm5
  
  rtg.comment "End of test0"
}

// CHECK-NEXT:    # End of test0
// CHECK-ALLOWED-NEXT:    # End of test0
