// RUN: rtgtool %s --seed=0 --unsupported-instructions=rtgtest.instr_a --emit-asm | FileCheck %s

// CHECK: \\ rtgtest.instr_a
// CHECK: .word 0x
%seq1 = rtg.sequence {
  %imm1 = arith.constant 4 : i32
  %imm2 = arith.constant 8 : i32
  rtgtest.instr_a %imm1, %imm2
} -> !rtg.sequence
