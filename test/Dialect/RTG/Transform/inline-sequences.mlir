// RUN: circt-opt --rtg-inline-sequences --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-NOT: rtg.sequence
rtg.sequence @seq0() {
  rtgtest.rv32i.ebreak
  rtgtest.rv32i.ebreak
}

rtg.sequence @seq1() {
  rtgtest.rv32i.ecall
  rtgtest.rv32i.ecall
}

// CHECK-LABEL: @inlineSequences
rtg.test @inlineSequences() {
  // CHECK-NEXT: rtgtest.rv32i.ecall
  // CHECK-NEXT: rtgtest.rv32i.ebreak
  // CHECK-NEXT: rtgtest.rv32i.ebreak
  // CHECK-NEXT: rtgtest.rv32i.ecall
  // CHECK-NEXT: }
  %0 = rtg.get_sequence @seq0 : !rtg.sequence
  %1 = rtg.randomize_sequence %0
  rtgtest.rv32i.ecall
  rtg.embed_sequence %1
  rtgtest.rv32i.ecall
}

// CHECK-LABEL: @interleaveSequences
rtg.test @interleaveSequences() {
  %0 = rtg.get_sequence @seq0 : !rtg.sequence
  %1 = rtg.get_sequence @seq1 : !rtg.sequence
  %2 = rtg.randomize_sequence %0
  %3 = rtg.randomize_sequence %1

  // CHECK-NEXT: rtgtest.rv32i.ebreak
  // CHECK-NEXT: rtgtest.rv32i.ebreak
  // CHECK-NEXT: rtgtest.rv32i.ecall
  // CHECK-NEXT: rtgtest.rv32i.ecall
  %4 = rtg.interleave_sequences %2, %3 batch 2
  rtg.embed_sequence %4

  // CHECK-NEXT: rtgtest.rv32i.ebreak
  // CHECK-NEXT: rtgtest.rv32i.ecall
  // CHECK-NEXT: rtgtest.rv32i.ebreak
  // CHECK-NEXT: rtgtest.rv32i.ecall
  %5 = rtg.interleave_sequences %2, %3
  rtg.embed_sequence %5

  // CHECK-NEXT: rtgtest.rv32i.ebreak
  // CHECK-NEXT: rtgtest.rv32i.ecall
  // CHECK-NEXT: rtgtest.rv32i.ecall
  // CHECK-NEXT: rtgtest.rv32i.ecall
  // CHECK-NEXT: rtgtest.rv32i.ebreak
  // CHECK-NEXT: rtgtest.rv32i.ecall
  %6 = rtg.interleave_sequences %2, %3
  %7 = rtg.interleave_sequences %6, %3
  rtg.embed_sequence %7

  // CHECK-NEXT: }
}

rtg.sequence @nested0() {
  %ra = rtg.fixed_reg #rtgtest.ra
  %sp = rtg.fixed_reg #rtgtest.s0
  %imm = rtg.constant #rtg.isa.immediate<12, 1>
  rtgtest.rv32i.jalr %ra, %sp, %imm
}

rtg.sequence @nested1() {
  %0 = rtg.get_sequence @nested0 : !rtg.sequence
  %1 = rtg.randomize_sequence %0
  rtg.embed_sequence %1
  %ra = rtg.fixed_reg #rtgtest.ra
  %sp = rtg.fixed_reg #rtgtest.sp
  %imm = rtg.constant #rtg.isa.immediate<12, 0>
  rtgtest.rv32i.jalr %ra, %sp, %imm
}

// CHECK-LABEL: @nestedSequences()
rtg.test @nestedSequences() {
  // CHECK-NEXT: [[RA0:%.+]] = rtg.fixed_reg #rtgtest.ra : !rtgtest.ireg
  // CHECK-NEXT: [[S0:%.+]] = rtg.fixed_reg #rtgtest.s0 : !rtgtest.ireg
  // CHECK-NEXT: [[IMM1:%.+]] = rtg.constant #rtg.isa.immediate<12, 1>
  // CHECK-NEXT: rtgtest.rv32i.jalr [[RA0]], [[S0]], [[IMM1]]
  // CHECK-NEXT: [[RA1:%.+]] = rtg.fixed_reg #rtgtest.ra : !rtgtest.ireg
  // CHECK-NEXT: [[SP:%.+]] = rtg.fixed_reg #rtgtest.sp : !rtgtest.ireg
  // CHECK-NEXT: [[IMM0:%.+]] = rtg.constant #rtg.isa.immediate<12, 0>
  // CHECK-NEXT: rtgtest.rv32i.jalr [[RA1]], [[SP]], [[IMM0]]
  %0 = rtg.get_sequence @nested1 : !rtg.sequence
  %1 = rtg.randomize_sequence %0
  rtg.embed_sequence %1
}

rtg.sequence @seqWithArgs(%imm: !rtg.isa.immediate<12>, %seq: !rtg.randomized_sequence) {
  %sp = rtg.fixed_reg #rtgtest.sp
  rtgtest.rv32i.jalr %sp, %sp, %imm
  rtg.embed_sequence %seq
}

// CHECK-LABEL: @substitutions
rtg.test @substitutions() {
  // CHECK-NEXT: [[IMM0:%.+]] = rtg.constant #rtg.isa.immediate<12, 0> : !rtg.isa.immediate<12>
  // CHECK-NEXT: [[SP:%.+]] = rtg.fixed_reg #rtgtest.sp : !rtgtest.ireg
  // CHECK-NEXT: rtgtest.rv32i.jalr [[SP]], [[SP]], [[IMM0]]
  // CHECK-NEXT: [[RA:%.+]] = rtg.fixed_reg #rtgtest.ra : !rtgtest.ireg
  // CHECK-NEXT: [[S0:%.+]] = rtg.fixed_reg #rtgtest.s0 : !rtgtest.ireg
  // CHECK-NEXT: [[IMM1:%.+]] = rtg.constant #rtg.isa.immediate<12, 1> : !rtg.isa.immediate<12>
  // CHECK-NEXT: rtgtest.rv32i.jalr [[RA]], [[S0]], [[IMM1]]
  %imm = rtg.constant #rtg.isa.immediate<12, 0>
  %0 = rtg.get_sequence @seqWithArgs : !rtg.sequence<!rtg.isa.immediate<12>, !rtg.randomized_sequence>
  %1 = rtg.get_sequence @nested0 : !rtg.sequence
  %2 = rtg.randomize_sequence %1
  %3 = rtg.substitute_sequence %0(%imm, %2) : !rtg.sequence<!rtg.isa.immediate<12>, !rtg.randomized_sequence>
  %4 = rtg.randomize_sequence %3
  rtg.embed_sequence %4
}

// -----

rtg.test @test0(seq = %seq : !rtg.randomized_sequence) {
  // expected-error @below {{sequence operand could not be resolved; it was likely produced by an op or block argument not supported by this pass}}
  rtg.embed_sequence %seq
}

// -----

rtg.test @test0(seq = %seq : !rtg.sequence) {
  // expected-error @below {{sequence operand could not be resolved; it was likely produced by an op or block argument not supported by this pass}}
  %0 = rtg.randomize_sequence %seq
  rtg.embed_sequence %0
}

// -----

rtg.test @test0(seq0 = %seq0 : !rtg.randomized_sequence, seq1 = %seq1 : !rtg.randomized_sequence) {
  // expected-error @below {{sequence operand #0 could not be resolved; it was likely produced by an op or block argument not supported by this pass}}
  %0 = rtg.interleave_sequences %seq0, %seq1
  rtg.embed_sequence %0
}
