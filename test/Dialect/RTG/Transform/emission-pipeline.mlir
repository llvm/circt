// RUN: circt-opt %s --rtg-emission-pipeline="split-output=false path=%t" | FileCheck %s && FileCheck %s --input-file=%t --check-prefix=ASM

// Test that the emission pipeline:
// 1. Inserts emit.file operations
// 2. Allocates virtual registers to physical registers
// 3. Creates rtg.segment operations
// 4. Lowers unique labels to concrete labels
// 5. Inlines tests into the file operations
// 6. Assembly is correctly emitted to the right file

// CHECK-NOT: rtg.test

// Test 1: emit.file operations
// CHECK-LABEL: emit.file "{{.*}}"

// CHECK-NOT: rtg.test

// Test 2: Virtual registers are allocated to physical registers
// CHECK-DAG: rtg.constant #rtgtest.{{[a-z0-9]+}} : !rtgtest.ireg
// CHECK-DAG: rtg.constant #rtgtest.{{[a-z0-9]+}} : !rtgtest.ireg

// CHECK-NOT: rtg.test

// Test 3: rtg.isa.segment operations are created
// CHECK: rtg.isa.segment text

// CHECK-NOT: rtg.test

// Test 4: Unique labels are lowered to concrete labels
// CHECK-DAG: [[L1:%.+]] = rtg.constant #rtg.isa.label<"unique_0">
// CHECK-DAG: [[L2:%.+]] = rtg.constant #rtg.isa.label<"unique_1">
// CHECK-DAG: rtg.label local [[L1]]
// CHECK-DAG: rtg.label local [[L2]]
// CHECK-DAG: rtg.label local [[L2]]
// CHECK-NOT: rtg.label_unique_decl

// CHECK-NOT: rtg.test

// Test 6: Assembly is emitted to the file
// ASM: .text
// ASM: # Begin of test 'test_labels'
// ASM: label:
// ASM: unique_0:
// ASM: unique_1:
// ASM: unique_1:
// ASM: # End of test 'test_labels'
// ASM: # Begin of test 'test_registers'
// ASM: jalr
// ASM: jalr
// ASM: # End of test 'test_registers'

rtg.test @test_labels() {
  %l0 = rtg.constant #rtg.isa.label<"label"> : !rtg.isa.label
  %str = rtg.constant "unique" : !rtg.string
  %l1 = rtg.label_unique_decl %str
  %l2 = rtg.label_unique_decl %str
  rtg.label local %l0
  rtg.label local %l1
  rtg.label local %l2
  rtg.label local %l2
}

rtg.test @test_registers() {
  %vreg0 = rtg.virtual_reg [#rtgtest.ra, #rtgtest.s0, #rtgtest.s1]
  %vreg1 = rtg.virtual_reg [#rtgtest.ra, #rtgtest.s0, #rtgtest.s1]
  %imm = rtg.constant #rtg.isa.immediate<12, 0> : !rtg.isa.immediate<12>
  rtgtest.rv32i.jalr %vreg0, %vreg1, %imm
  rtgtest.rv32i.jalr %vreg1, %vreg0, %imm
}

