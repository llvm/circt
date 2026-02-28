// RUN: circt-opt %s --rtg-randomization-pipeline=seed=0 | FileCheck %s

rtg.target @target : !rtg.dict<mem_blk: !rtg.isa.memory_block<32>> {
  %0 = rtg.isa.memory_block_declare [0 - 31] : !rtg.isa.memory_block<32>
  rtg.yield %0 : !rtg.isa.memory_block<32>
}

// Test that the randomization pipeline:
// 1. Elaborates tests (creates instances with target names)
// 2. Inlines sequences
// 3. Allocates memories to immediates (default behavior)
// 4. Does NOT insert emit.file operations
// 5. Does NOT lower unique labels

// CHECK-NOT: emit.file

// Test 1: Elaboration - creates test instances with target names
// CHECK-LABEL: rtg.test @test_memories_target
// CHECK-SAME: template "test_memories" target @target
// CHECK: [[IMM:%.+]] = rtg.constant #rtg.isa.immediate<32, 0> : !rtg.isa.immediate<32>
// CHECK: rtgtest.rv32i.la {{%.+}}, [[IMM]] : !rtg.isa.immediate<32>
rtg.test @test_memories(mem_blk = %mem_blk: !rtg.isa.memory_block<32>) {
  %idx4 = index.constant 4
  %0 = rtg.isa.memory_alloc %mem_blk, %idx4, %idx4 : !rtg.isa.memory_block<32>
  %reg = rtg.virtual_reg [#rtgtest.t0, #rtgtest.t1, #rtgtest.t2]
  rtgtest.rv32i.la %reg, %0 : !rtg.isa.memory<32>
}

// Test 2: Sequence inlining - sequences are inlined into tests
// CHECK-LABEL: rtg.test @test_sequences
// CHECK-NOT: rtg.get_sequence
// CHECK-NOT: rtg.randomize_sequence
// CHECK-NOT: rtg.embed_sequence
// CHECK: [[REG:%.+]] = rtg.constant #rtgtest.t0
// CHECK: [[IMM:%.+]] = rtg.constant #rtg.isa.immediate<12, 42>
// CHECK: rtgtest.rv32i.addi [[REG]], [[REG]], [[IMM]]
rtg.sequence @seq() {
  %reg = rtg.constant #rtgtest.t0 : !rtgtest.ireg
  %imm = rtg.constant #rtg.isa.immediate<12, 42> : !rtg.isa.immediate<12>
  "rtgtest.rv32i.addi"(%reg, %reg, %imm) : (!rtgtest.ireg, !rtgtest.ireg, !rtg.isa.immediate<12>) -> ()
}

rtg.test @test_sequences() {
  %0 = rtg.get_sequence @seq : !rtg.sequence
  %1 = rtg.randomize_sequence %0
  rtg.embed_sequence %1
}

// Test 3: Unique labels are not lowered in the randomization pipeline
// CHECK-LABEL: rtg.test @test_unique_labels_target
// CHECK: [[L1:%.+]] = rtg.label_unique_decl
// CHECK: rtg.label local [[L1]]
// CHECK: [[L2:%.+]] = rtg.label_unique_decl
// CHECK: rtg.label local [[L2]]
rtg.test @test_unique_labels() {
  %str = rtg.constant "label" : !rtg.string
  %l1 = rtg.label_unique_decl %str
  %l2 = rtg.label_unique_decl %str
  rtg.label local %l1
  rtg.label local %l2
}

// CHECK-NOT: emit.file
