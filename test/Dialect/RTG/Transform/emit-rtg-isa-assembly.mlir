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
}

// CHECK-EMPTY:
// CHECK-NEXT:# End of test0
// CHECK-ALLOWED-EMPTY:
// CHECK-ALLOWED-NEXT:# End of test0
