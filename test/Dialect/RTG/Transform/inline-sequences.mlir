// RUN: circt-opt --rtg-inline-sequences=fail-on-remaining=true --split-input-file --verify-diagnostics --mlir-print-debuginfo --mlir-print-local-scope %s | FileCheck %s

rtg.sequence @seq0() {
  rtgtest.zero_operand_instr {a}
  rtgtest.zero_operand_instr {a}
}

rtg.sequence @seq1() {
  rtgtest.zero_operand_instr {b}
  rtgtest.zero_operand_instr {b}
}

// CHECK-LABEL: @inlineSequences
rtg.test @inlineSequences() {
  // CHECK-NEXT: rtgtest.zero_operand_instr {b}
  // CHECK-NEXT: rtgtest.zero_operand_instr {a} loc("loc_0")
  // CHECK-NEXT: rtgtest.zero_operand_instr {a} loc("loc_0")
  // CHECK-NEXT: rtgtest.zero_operand_instr {b}
  // CHECK-NEXT: }
  %0 = rtg.get_sequence @seq0 : !rtg.sequence
  %1 = rtg.randomize_sequence %0
  rtgtest.zero_operand_instr {b}
  rtg.embed_sequence %1 loc("loc_0")
  rtgtest.zero_operand_instr {b}
}

// CHECK-LABEL: @interleaveSequences
rtg.test @interleaveSequences() {
  %0 = rtg.get_sequence @seq0 : !rtg.sequence
  %1 = rtg.get_sequence @seq1 : !rtg.sequence
  %2 = rtg.randomize_sequence %0
  %3 = rtg.randomize_sequence %1

  // CHECK-NEXT: rtgtest.zero_operand_instr {a}
  // CHECK-NEXT: rtgtest.zero_operand_instr {a}
  // CHECK-NEXT: rtgtest.zero_operand_instr {b}
  // CHECK-NEXT: rtgtest.zero_operand_instr {b}
  %4 = rtg.interleave_sequences %2, %3 batch 2
  rtg.embed_sequence %4

  // CHECK-NEXT: rtgtest.zero_operand_instr {a}
  // CHECK-NEXT: rtgtest.zero_operand_instr {b}
  // CHECK-NEXT: rtgtest.zero_operand_instr {a}
  // CHECK-NEXT: rtgtest.zero_operand_instr {b}
  %5 = rtg.interleave_sequences %2, %3
  rtg.embed_sequence %5

  // CHECK-NEXT: rtgtest.zero_operand_instr {a}
  // CHECK-NEXT: rtgtest.zero_operand_instr {b}
  // CHECK-NEXT: rtgtest.zero_operand_instr {b}
  // CHECK-NEXT: rtgtest.zero_operand_instr {b}
  // CHECK-NEXT: rtgtest.zero_operand_instr {a}
  // CHECK-NEXT: rtgtest.zero_operand_instr {b}
  %6 = rtg.interleave_sequences %2, %3
  %7 = rtg.interleave_sequences %6, %3
  rtg.embed_sequence %7

  // CHECK-NEXT: }
}

rtg.sequence @nested0() {
  %ra = rtg.constant #rtgtest.ra
  %sp = rtg.constant #rtgtest.s0
  rtgtest.two_register_instr %ra, %sp
}

rtg.sequence @nested1() {
  %0 = rtg.get_sequence @nested0 : !rtg.sequence
  %1 = rtg.randomize_sequence %0
  rtg.embed_sequence %1
  %ra = rtg.constant #rtgtest.ra
  %sp = rtg.constant #rtgtest.sp
  rtgtest.two_register_instr %ra, %sp
}

// CHECK-LABEL: @nestedSequences()
rtg.test @nestedSequences() {
  // CHECK-NEXT: [[RA0:%.+]] = rtg.constant #rtgtest.ra : !rtgtest.ireg
  // CHECK-NEXT: [[S0:%.+]] = rtg.constant #rtgtest.s0 : !rtgtest.ireg
  // CHECK-NEXT: rtgtest.two_register_instr [[RA0]], [[S0]]
  // CHECK-NEXT: [[RA1:%.+]] = rtg.constant #rtgtest.ra : !rtgtest.ireg
  // CHECK-NEXT: [[SP:%.+]] = rtg.constant #rtgtest.sp : !rtgtest.ireg
  // CHECK-NEXT: rtgtest.two_register_instr [[RA1]], [[SP]]
  %0 = rtg.get_sequence @nested1 : !rtg.sequence
  %1 = rtg.randomize_sequence %0
  rtg.embed_sequence %1
}

rtg.sequence @seqWithArgs(%seq: !rtg.randomized_sequence) {
  %sp = rtg.constant #rtgtest.sp
  rtgtest.two_register_instr %sp, %sp
  rtg.embed_sequence %seq
}

// CHECK-LABEL: @substitutions
rtg.test @substitutions() {
  // CHECK-NEXT: [[SP:%.+]] = rtg.constant #rtgtest.sp : !rtgtest.ireg
  // CHECK-NEXT: rtgtest.two_register_instr [[SP]], [[SP]]
  // CHECK-NEXT: [[RA:%.+]] = rtg.constant #rtgtest.ra : !rtgtest.ireg
  // CHECK-NEXT: [[S0:%.+]] = rtg.constant #rtgtest.s0 : !rtgtest.ireg
  // CHECK-NEXT: rtgtest.two_register_instr [[RA]], [[S0]]
  %0 = rtg.get_sequence @seqWithArgs : !rtg.sequence<!rtg.randomized_sequence>
  %1 = rtg.get_sequence @nested0 : !rtg.sequence
  %2 = rtg.randomize_sequence %1
  %3 = rtg.substitute_sequence %0(%2) : !rtg.sequence<!rtg.randomized_sequence>
  %4 = rtg.randomize_sequence %3
  rtg.embed_sequence %4
}

// CHECK-LABEL: @nestedRegion
rtg.test @nestedRegion() {
  // CHECK-NEXT: scf.execute_region {
  scf.execute_region {
    // CHECK-NEXT: rtgtest.zero_operand_instr {a}
    // CHECK-NEXT: rtgtest.zero_operand_instr {a}
    // CHECK-NEXT: scf.yield
    %0 = rtg.get_sequence @seq0 : !rtg.sequence
    %1 = rtg.randomize_sequence %0
    rtg.embed_sequence %1
    scf.yield
  }
}

// -----

rtg.sequence @seq() {}

rtg.test @test0(seq = %seq : !rtg.randomized_sequence) {
  // expected-error @below {{sequence operand could not be resolved; it was likely produced by an op or block argument not supported by this pass}}
  rtg.embed_sequence %seq
}

// -----

rtg.sequence @seq() {}

rtg.test @test0(seq = %seq : !rtg.sequence) {
  // expected-error @below {{sequence operand could not be resolved; it was likely produced by an op or block argument not supported by this pass}}
  %0 = rtg.randomize_sequence %seq
  rtg.embed_sequence %0
}

// -----

rtg.sequence @seq() {}

rtg.test @test0(seq0 = %seq0 : !rtg.randomized_sequence, seq1 = %seq1 : !rtg.randomized_sequence) {
  // expected-error @below {{sequence operand could not be resolved; it was likely produced by an op or block argument not supported by this pass}}
  %0 = rtg.interleave_sequences %seq0, %seq1
  rtg.embed_sequence %0
}
