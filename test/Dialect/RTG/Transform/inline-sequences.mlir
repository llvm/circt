// RUN: circt-opt --rtg-inline-sequences --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-NOT: rtg.sequence
rtg.sequence @seq0() {
  rtgtest.rv32i.ebreak
}

// CHECK-LABEL: @interleaveSequences
rtg.test @interleaveSequences() {
  // CHECK-NEXT: rtgtest.rv32i.ecall
  // CHECK-NEXT: rtgtest.rv32i.ebreak
  // CHECK-NEXT: rtgtest.rv32i.ecall
  // CHECK-NEXT: }

  %0 = rtg.get_sequence @seq0 : !rtg.sequence
  %1 = rtg.randomize_sequence %0
  rtgtest.rv32i.ecall
  rtg.embed_sequence %1
  rtgtest.rv32i.ecall
}

// -----

rtg.test @test0(seq = %seq : !rtg.randomized_sequence) {
  // expected-error @below {{sequence operand not directly defined by 'rtg.randomize_sequence' op}}
  rtg.embed_sequence %seq
}

// -----

rtg.test @test0(seq = %seq : !rtg.sequence) {
  // expected-error @below {{sequence operand not directly defined by 'rtg.get_sequence' op}}
  %0 = rtg.randomize_sequence %seq
  rtg.embed_sequence %0
}
