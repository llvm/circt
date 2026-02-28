// RUN: circt-opt --rtg-inline-sequences=fail-on-remaining=false %s | FileCheck %s

// CHECK-LABEL: @test0
rtg.test @test0(seq = %seq : !rtg.randomized_sequence) {
  // CHECK-NEXT: rtg.embed_sequence %seq
  rtg.embed_sequence %seq
}

// CHECK-LABEL: @test1
rtg.test @test1(seq = %seq : !rtg.sequence) {
  // CHECK-NEXT: [[SEQ:%.+]] = rtg.randomize_sequence %seq
  // CHECK-NEXT: rtg.embed_sequence [[SEQ]]
  %0 = rtg.randomize_sequence %seq
  rtg.embed_sequence %0
}

// CHECK-LABEL: @test2
rtg.test @test2(seq0 = %seq0 : !rtg.randomized_sequence, seq1 = %seq1 : !rtg.randomized_sequence) {
  // CHECK-NEXT: [[SEQ:%.+]] = rtg.interleave_sequences %seq0, %seq1
  // CHECK-NEXT: rtg.embed_sequence [[SEQ]]
  %0 = rtg.interleave_sequences %seq0, %seq1
  rtg.embed_sequence %0
}
