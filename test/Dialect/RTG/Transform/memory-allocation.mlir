// RUN: circt-opt --rtg-memory-allocation --split-input-file --verify-diagnostics %s | FileCheck %s

// Test proper alignment computation.
rtg.target @target : !rtg.dict<mem_blk: !rtg.isa.memory_block<32>> {
  %0 = rtg.isa.memory_block_declare [0 - 31] : !rtg.isa.memory_block<32>
  rtg.yield %0 : !rtg.isa.memory_block<32>
}

// CHECK-LABEL: rtg.test @test
rtg.test @test(mem_blk = %mem_blk: !rtg.isa.memory_block<32>) target @target {
  // CHECK: [[IMM0:%.+]] = rtg.constant #rtg.isa.immediate<32, 0>
  // CHECK: [[REG:%.+]] = rtg.constant
  // CHECK: rtgtest.rv32i.la [[REG]], [[IMM0]] : !rtg.isa.immediate<32>
  %idx4 = index.constant 4
  %idx7 = index.constant 7
  %0 = rtg.isa.memory_alloc %mem_blk, %idx7, %idx4 : !rtg.isa.memory_block<32>
  %reg = rtg.constant #rtgtest.t0
  rtgtest.rv32i.la %reg, %0 : !rtg.isa.memory<32>

  // CHECK: [[IMM1:%.+]] = rtg.constant #rtg.isa.immediate<32, 8>
  // CHECK: rtgtest.rv32i.la [[REG]], [[IMM1]] : !rtg.isa.immediate<32>
  %1 = rtg.isa.memory_alloc %mem_blk, %idx7, %idx4 : !rtg.isa.memory_block<32>
  rtgtest.rv32i.la %reg, %1 : !rtg.isa.memory<32>
}

// -----

rtg.target @target : !rtg.dict<mem_blk: !rtg.isa.memory_block<32>> {
  %0 = rtg.isa.memory_block_declare [0 - 31] : !rtg.isa.memory_block<32>
  rtg.yield %0 : !rtg.isa.memory_block<32>
}

func.func @passthrough(%mem_blk: !rtg.isa.memory_block<32>) -> !rtg.isa.memory_block<32> {
  return %mem_blk : !rtg.isa.memory_block<32>
}

rtg.test @test_missing_block(mem_blk = %mem_blk: !rtg.isa.memory_block<32>) target @target {
  %idx4 = index.constant 4
  %idx7 = index.constant 7
  %mem_block_unknown = func.call @passthrough(%mem_blk) : (!rtg.isa.memory_block<32>) -> !rtg.isa.memory_block<32>
  // expected-error @below {{memory block not found}}
  %0 = rtg.isa.memory_alloc %mem_block_unknown, %idx7, %idx4 : !rtg.isa.memory_block<32>
  %reg = rtg.constant #rtgtest.t0
  rtgtest.rv32i.la %reg, %0 : !rtg.isa.memory<32>
}

// -----

rtg.target @target_unknown_size : !rtg.dict<mem_blk: !rtg.isa.memory_block<32>> {
  %0 = rtg.isa.memory_block_declare [0 - 31] : !rtg.isa.memory_block<32>
  rtg.yield %0 : !rtg.isa.memory_block<32>
}

func.func @const() -> index {
  %0 = index.constant 0
  return %0 : index
}

rtg.test @test_unknown_size(mem_blk = %mem_blk: !rtg.isa.memory_block<32>) target @target_unknown_size {
  %unknown = func.call @const() : () -> index
  %idx4 = index.constant 4
  // expected-error @below {{could not determine memory allocation size}}
  %0 = rtg.isa.memory_alloc %mem_blk, %unknown, %idx4 : !rtg.isa.memory_block<32>
  %reg = rtg.constant #rtgtest.t0
  rtgtest.rv32i.la %reg, %0 : !rtg.isa.memory<32>
}

// -----

rtg.target @target_unknown_align : !rtg.dict<mem_blk: !rtg.isa.memory_block<32>> {
  %0 = rtg.isa.memory_block_declare [0 - 31] : !rtg.isa.memory_block<32>
  rtg.yield %0 : !rtg.isa.memory_block<32>
}

func.func @const() -> index {
  %0 = index.constant 0
  return %0 : index
}

rtg.test @test_unknown_align(mem_blk = %mem_blk: !rtg.isa.memory_block<32>) target @target_unknown_align {
  %idx7 = index.constant 7
  %unknown = func.call @const() : () -> index
  // expected-error @below {{could not determine memory allocation alignment}}
  %0 = rtg.isa.memory_alloc %mem_blk, %idx7, %unknown : !rtg.isa.memory_block<32>
  %reg = rtg.constant #rtgtest.t0
  rtgtest.rv32i.la %reg, %0 : !rtg.isa.memory<32>
}

// -----

rtg.target @target_zero_size : !rtg.dict<mem_blk: !rtg.isa.memory_block<32>> {
  %0 = rtg.isa.memory_block_declare [0 - 31] : !rtg.isa.memory_block<32>
  rtg.yield %0 : !rtg.isa.memory_block<32>
}

rtg.test @test_zero_size(mem_blk = %mem_blk: !rtg.isa.memory_block<32>) target @target_zero_size {
  %idx0 = index.constant 0
  %idx4 = index.constant 4
  // expected-error @below {{memory allocation size must be greater than zero (was 0)}}
  %0 = rtg.isa.memory_alloc %mem_blk, %idx0, %idx4 : !rtg.isa.memory_block<32>
  %reg = rtg.constant #rtgtest.t0
  rtgtest.rv32i.la %reg, %0 : !rtg.isa.memory<32>
}

// -----

rtg.target @target_non_pow2_align : !rtg.dict<mem_blk: !rtg.isa.memory_block<32>> {
  %0 = rtg.isa.memory_block_declare [0 - 31] : !rtg.isa.memory_block<32>
  rtg.yield %0 : !rtg.isa.memory_block<32>
}

rtg.test @test_non_pow2_align(mem_blk = %mem_blk: !rtg.isa.memory_block<32>) target @target_non_pow2_align {
  %idx7 = index.constant 7
  %idx3 = index.constant 3
  // expected-error @below {{memory allocation alignment must be a power of two (was 3)}}
  %0 = rtg.isa.memory_alloc %mem_blk, %idx7, %idx3 : !rtg.isa.memory_block<32>
  %reg = rtg.constant #rtgtest.t0
  rtgtest.rv32i.la %reg, %0 : !rtg.isa.memory<32>
}

// -----

rtg.target @target_exceed_size : !rtg.dict<mem_blk: !rtg.isa.memory_block<32>> {
  %0 = rtg.isa.memory_block_declare [0 - 7] : !rtg.isa.memory_block<32>
  rtg.yield %0 : !rtg.isa.memory_block<32>
}

rtg.test @test_exceed_size(mem_blk = %mem_blk: !rtg.isa.memory_block<32>) target @target_exceed_size {
  %idx16 = index.constant 16
  %idx4 = index.constant 4
  // expected-error @below {{memory block not large enough to fit all allocations}}
  %0 = rtg.isa.memory_alloc %mem_blk, %idx16, %idx4 : !rtg.isa.memory_block<32>
  %reg = rtg.constant #rtgtest.t0
  rtgtest.rv32i.la %reg, %0 : !rtg.isa.memory<32>
}

// -----

rtg.target @target_truncate_error : !rtg.dict<mem_blk: !rtg.isa.memory_block<8>> {
  %0 = rtg.isa.memory_block_declare [0 - 31] : !rtg.isa.memory_block<8>
  rtg.yield %0 : !rtg.isa.memory_block<8>
}

rtg.test @test_truncate_error(mem_blk = %mem_blk: !rtg.isa.memory_block<8>) target @target_truncate_error {
  %idx256 = index.constant 256
  %idx4 = index.constant 4
  // expected-error @below {{cannot truncate APInt because value is too big to fit}}
  %0 = rtg.isa.memory_alloc %mem_blk, %idx256, %idx4 : !rtg.isa.memory_block<8>
  %reg = rtg.constant #rtgtest.t0
  rtgtest.rv32i.la %reg, %0 : !rtg.isa.memory<8>
}

// -----

// Test for proper matching of target entries to test arguments by name.
rtg.target @target_multiple_entries : !rtg.dict<mem_a: !rtg.isa.memory_block<32>, mem_b: !rtg.isa.memory_block<32>, mem_c: !rtg.isa.memory_block<32>> {
  %mem_a = rtg.isa.memory_block_declare [0 - 31] : !rtg.isa.memory_block<32>
  %mem_b = rtg.isa.memory_block_declare [100 - 131] : !rtg.isa.memory_block<32>
  %mem_c = rtg.isa.memory_block_declare [200 - 231] : !rtg.isa.memory_block<32>

  rtg.yield %mem_a, %mem_b, %mem_c : !rtg.isa.memory_block<32>, !rtg.isa.memory_block<32>, !rtg.isa.memory_block<32>
}

// CHECK-LABEL: rtg.test @test_entry_matching
rtg.test @test_entry_matching(
  mem_a = %mem_a: !rtg.isa.memory_block<32>,
  mem_c = %mem_c: !rtg.isa.memory_block<32>
) target @target_multiple_entries {
  %idx8 = index.constant 8
  %idx4 = index.constant 4
  %reg = rtg.constant #rtgtest.t0

  // CHECK: [[MEM_A:%.+]] = rtg.constant #rtg.isa.immediate<32, 0>
  %0 = rtg.isa.memory_alloc %mem_a, %idx8, %idx4 : !rtg.isa.memory_block<32>
  rtgtest.rv32i.la %reg, %0 : !rtg.isa.memory<32>

  // CHECK: [[MEM_C:%.+]] = rtg.constant #rtg.isa.immediate<32, 200>
  %2 = rtg.isa.memory_alloc %mem_c, %idx8, %idx4 : !rtg.isa.memory_block<32>
  rtgtest.rv32i.la %reg, %2 : !rtg.isa.memory<32>
}
