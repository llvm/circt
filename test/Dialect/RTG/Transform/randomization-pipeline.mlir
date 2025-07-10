// RUN: circt-opt %s --rtg-randomization-pipeline=seed=0 | FileCheck %s

rtg.target @target : !rtg.dict<mem_blk: !rtg.isa.memory_block<32>> {
  %0 = rtg.isa.memory_block_declare [0 - 31] : !rtg.isa.memory_block<32>
  rtg.yield %0 : !rtg.isa.memory_block<32>
}

// CHECK-LABEL: rtg.test @test_target
rtg.test @test(mem_blk = %mem_blk: !rtg.isa.memory_block<32>) {
  // CHECK-NEXT: [[REG:%.+]] = rtg.fixed_reg
  // CHECK-NEXT: [[IMM:%.+]] = rtg.constant #rtg.isa.immediate
  // CHECK-NEXT: rtgtest.rv32i.la [[REG]], [[IMM]] :
  %idx4 = index.constant 4
  %0 = rtg.isa.memory_alloc %mem_blk, %idx4, %idx4 : !rtg.isa.memory_block<32>
  %reg = rtg.virtual_reg [#rtgtest.t0, #rtgtest.t1, #rtgtest.t2]
  rtgtest.rv32i.la %reg, %0 : !rtg.isa.memory<32>
}
