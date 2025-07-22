// RUN: circt-opt --canonicalize %s | FileCheck %s

// CHECK-LABEL: @interleaveSequences
rtg.test @interleaveSequences(seq0 = %seq0: !rtg.randomized_sequence) {
  // CHECK-NEXT: rtg.embed_sequence %seq0
  %0 = rtg.interleave_sequences %seq0
  rtg.embed_sequence %0
}

// CHECK-LABEL: @immediates
rtg.target @immediates : !rtg.dict<imm0: !rtg.isa.immediate<64>, imm1: !rtg.isa.immediate<2>> {
  %0 = rtg.constant #rtg.isa.immediate<32, -1>
  %1 = rtg.constant #rtg.isa.immediate<32, 0>
  %3 = rtg.isa.concat_immediate %1, %0 : !rtg.isa.immediate<32>, !rtg.isa.immediate<32>
  %4 = rtg.isa.slice_immediate %3 from 31 : !rtg.isa.immediate<64> -> !rtg.isa.immediate<2>
  
  // CHECK-NEXT: [[V0:%.+]] = rtg.constant #rtg.isa.immediate<64, 4294967295>
  // CHECK-NEXT: [[V1:%.+]] = rtg.constant #rtg.isa.immediate<2, 1>
  // CHECK-NEXT: rtg.yield [[V0]], [[V1]] :
  rtg.yield %3, %4 : !rtg.isa.immediate<64>, !rtg.isa.immediate<2>
}
