// RUN: split-file %s %t

//--- test0.mlir
// RUN: circt-opt %t/test0.mlir --rtg-embed-validation-values=filename=%S/validation-values-0.txt | FileCheck %s --check-prefix=CHECK-TEST0

// CHECK-TEST0-LABEL: rtg.test @embed_value
rtg.test @embed_value() {
  // CHECK-TEST0-NEXT: [[REG:%.+]] = rtg.fixed_reg #rtgtest.t0
  // CHECK-TEST0-NEXT: [[V0:%.+]] = rtg.constant #rtg.isa.immediate<32, 8192>
  // CHECK-TEST0-NEXT: rtgtest.rv32i.lui [[REG]], [[V0]] :
  // CHECK-TEST0-NEXT: }
  %reg = rtg.fixed_reg #rtgtest.t0
  %imm = rtg.constant #rtg.isa.immediate<32, 0x1000>
  %exp_val = rtg.validate %reg, %imm, "id1" : !rtgtest.ireg -> !rtg.isa.immediate<32>
  rtgtest.rv32i.lui %reg, %exp_val : !rtg.isa.immediate<32>
}

//--- test1.mlir
// RUN: not circt-opt %t/test1.mlir --rtg-embed-validation-values=filename=%S/validation-values-0.txt 2>&1 | FileCheck %s --check-prefix=CHECK-TEST1

// CHECK-TEST1: validation-values-0.txt:1:4: error: cannot parse value of type '!rtg.isa.immediate<2>' from string '0x2000'
rtg.test @value_parsing_error() {
  %reg = rtg.fixed_reg #rtgtest.t0
  %imm = rtg.constant #rtg.isa.immediate<2, 0>
  %exp_val = rtg.validate %reg, %imm, "id1" : !rtgtest.ireg -> !rtg.isa.immediate<2>
}

//--- test2.mlir
// RUN: not circt-opt %t/test2.mlir --rtg-embed-validation-values=filename=%S/validation-values-1.txt 2>&1 | FileCheck %s --check-prefix=CHECK-TEST2

// CHECK-TEST2: validation-values-1.txt:2:1: error: duplicate ID in input file: "id1"
rtg.test @duplicate_id_in_file_error() {
  %reg = rtg.fixed_reg #rtgtest.t0
  %imm = rtg.constant #rtg.isa.immediate<32, 0x1000>
  %exp_val = rtg.validate %reg, %imm, "id1" : !rtgtest.ireg -> !rtg.isa.immediate<32>
  rtgtest.rv32i.lui %reg, %exp_val : !rtg.isa.immediate<32>
}

//--- test3.mlir
// RUN: not circt-opt %t/test3.mlir --rtg-embed-validation-values=filename=%S/validation-values-2.txt 2>&1 | FileCheck %s --check-prefix=CHECK-TEST3

// CHECK-TEST3: validation-values-2.txt:2:4: error: no value for ID 'id0'
rtg.test @no_value_for_id_error() {
  %reg = rtg.fixed_reg #rtgtest.t0
  %imm = rtg.constant #rtg.isa.immediate<32, 0x1000>
  %exp_val = rtg.validate %reg, %imm, "id1" : !rtgtest.ireg -> !rtg.isa.immediate<32>
  rtgtest.rv32i.lui %reg, %exp_val : !rtg.isa.immediate<32>
}

//--- test4.mlir
// RUN: circt-opt %t/test4.mlir --rtg-embed-validation-values=filename=%S/validation-values-0.txt --verify-diagnostics

rtg.test @duplicate_validate_id_error() {
  %reg = rtg.fixed_reg #rtgtest.t0
  %imm = rtg.constant #rtg.isa.immediate<32, 0x1000>
  %exp_val = rtg.validate %reg, %imm, "id1" : !rtgtest.ireg -> !rtg.isa.immediate<32>
  // expected-error @below {{at least two validate ops have the same ID: "id1"}}
  %exp_val2 = rtg.validate %reg, %imm, "id1" : !rtgtest.ireg -> !rtg.isa.immediate<32>
  rtgtest.rv32i.lui %reg, %exp_val : !rtg.isa.immediate<32>
  rtgtest.rv32i.lui %reg, %exp_val2 : !rtg.isa.immediate<32>
}

//--- test5.mlir
// RUN: circt-opt %t/test5.mlir --rtg-embed-validation-values=filename=%S/validation-values-0.txt | FileCheck %s --check-prefix=CHECK-TEST5

// CHECK-TEST5-LABEL: rtg.test @unmached_validate_stays
rtg.test @unmached_validate_stays() {
  // CHECK-TEST5-NEXT: [[REG:%.+]] = rtg.fixed_reg #rtgtest.t0
  // CHECK-TEST5-NEXT: [[IMM:%.+]] = rtg.constant #rtg.isa.immediate<32, 4096>
  // CHECK-TEST5-NEXT: [[VAL:%.+]] = rtg.validate [[REG]], [[IMM]], "id10" : !rtgtest.ireg -> !rtg.isa.immediate<32>
  // CHECK-TEST5-NEXT: rtgtest.rv32i.lui [[REG]], [[VAL]] : !rtg.isa.immediate<32>
  %reg = rtg.fixed_reg #rtgtest.t0
  %imm = rtg.constant #rtg.isa.immediate<32, 0x1000>
  %exp_val = rtg.validate %reg, %imm, "id10" : !rtgtest.ireg -> !rtg.isa.immediate<32>
  rtgtest.rv32i.lui %reg, %exp_val : !rtg.isa.immediate<32>
}
