// RUN: split-file %s %t

//--- test0.mlir
// RUN: circt-opt %t/test0.mlir --rtg-embed-validation-values=filename=%S/validation-values-0.txt | FileCheck %s --check-prefix=CHECK-TEST0

// CHECK-TEST0-LABEL: rtg.test @embed_value
rtg.test @embed_value() {
  // CHECK-TEST0-NEXT: [[REG:%.+]] = rtg.constant #rtgtest.t0
  // CHECK-TEST0-NEXT: [[V0:%.+]] = rtg.constant 8192 : i32
  // CHECK-TEST0-NEXT: rtgtest.lui [[REG]], [[V0]] :
  // CHECK-TEST0-NEXT: }
  %reg = rtg.constant #rtgtest.t0
  %imm = rtg.constant 0x1000 : i32
  %exp_val = rtg.validate %reg, %imm, "id1" : !rtgtest.ireg -> i32
  rtgtest.lui %reg, %exp_val : i32
}

//--- test1.mlir
// RUN: not circt-opt %t/test1.mlir --rtg-embed-validation-values=filename=%S/validation-values-0.txt 2>&1 | FileCheck %s --check-prefix=CHECK-TEST1

// CHECK-TEST1: validation-values-0.txt:1:4: error: cannot parse value of type 'i2' from string '0x2000'
rtg.test @value_parsing_error() {
  %reg = rtg.constant #rtgtest.t0
  %imm = rtg.constant 0 : i2
  %exp_val = rtg.validate %reg, %imm, "id1" : !rtgtest.ireg -> i2
}

//--- test2.mlir
// RUN: not circt-opt %t/test2.mlir --rtg-embed-validation-values=filename=%S/validation-values-1.txt 2>&1 | FileCheck %s --check-prefix=CHECK-TEST2

// CHECK-TEST2: validation-values-1.txt:2:1: error: duplicate ID in input file: "id1"
rtg.test @duplicate_id_in_file_error() {
  %reg = rtg.constant #rtgtest.t0
  %imm = rtg.constant 0x1000 : i32
  %exp_val = rtg.validate %reg, %imm, "id1" : !rtgtest.ireg -> i32
  rtgtest.lui %reg, %exp_val : i32
}

//--- test3.mlir
// RUN: not circt-opt %t/test3.mlir --rtg-embed-validation-values=filename=%S/validation-values-2.txt 2>&1 | FileCheck %s --check-prefix=CHECK-TEST3

// CHECK-TEST3: validation-values-2.txt:2:4: error: no value for ID 'id0'
rtg.test @no_value_for_id_error() {
  %reg = rtg.constant #rtgtest.t0
  %imm = rtg.constant 0x1000 : i32
  %exp_val = rtg.validate %reg, %imm, "id1" : !rtgtest.ireg -> i32
  rtgtest.lui %reg, %exp_val : i32
}

//--- test4.mlir
// RUN: circt-opt %t/test4.mlir --rtg-embed-validation-values=filename=%S/validation-values-0.txt --verify-diagnostics

rtg.test @duplicate_validate_id_error() {
  %reg = rtg.constant #rtgtest.t0
  %imm = rtg.constant 0x1000 : i32
  %exp_val = rtg.validate %reg, %imm, "id1" : !rtgtest.ireg -> i32
  // expected-error @below {{at least two validate ops have the same ID: "id1"}}
  %exp_val2 = rtg.validate %reg, %imm, "id1" : !rtgtest.ireg -> i32
  rtgtest.lui %reg, %exp_val : i32
  rtgtest.lui %reg, %exp_val2 : i32
}

//--- test5.mlir
// RUN: circt-opt %t/test5.mlir --rtg-embed-validation-values=filename=%S/validation-values-0.txt | FileCheck %s --check-prefix=CHECK-TEST5

// CHECK-TEST5-LABEL: rtg.test @unmached_validate_stays
rtg.test @unmached_validate_stays() {
  // CHECK-TEST5-NEXT: [[REG:%.+]] = rtg.constant #rtgtest.t0
  // CHECK-TEST5-NEXT: [[IMM:%.+]] = rtg.constant 4096 : i32
  // CHECK-TEST5-NEXT: [[VAL:%.+]] = rtg.validate [[REG]], [[IMM]], "id10" : !rtgtest.ireg -> i32
  // CHECK-TEST5-NEXT: rtgtest.lui [[REG]], [[VAL]] : i32
  %reg = rtg.constant #rtgtest.t0
  %imm = rtg.constant 0x1000 : i32
  %exp_val = rtg.validate %reg, %imm, "id10" : !rtgtest.ireg -> i32
  rtgtest.lui %reg, %exp_val : i32
}

//--- test6.mlir
// RUN: circt-opt %t/test6.mlir --rtg-embed-validation-values=filename=%S/validation-values-0.txt | FileCheck %s --check-prefix=CHECK-TEST6

// CHECK-TEST6-LABEL: rtg.test @embed_value
rtg.test @embed_value() {
  // CHECK-TEST6-NEXT: [[REG:%.+]] = rtg.constant #rtgtest.t0
  // CHECK-TEST6-NEXT: [[V1:%.+]] = rtg.constant 4 : i32
  // CHECK-TEST6-NEXT: [[V2:%.+]] = rtg.constant 5 : i32
  // CHECK-TEST6-NEXT: [[V0:%.+]] = rtg.constant 8192 : i32
  // CHECK-TEST6-NEXT: rtgtest.lui [[REG]], [[V0]] :
  // CHECK-TEST6-NEXT: rtgtest.lui [[REG]], [[V1]] :
  // CHECK-TEST6-NEXT: rtgtest.lui [[REG]], [[V2]] :
  // CHECK-TEST6-NEXT: }
  %0 = rtg.constant #rtgtest.t0
  %1 = rtg.constant 1 : i32
  %2 = rtg.constant 2 : i32
  %3 = rtg.constant 3 : i32
  %4 = rtg.constant 4 : i32
  %5 = rtg.constant 5 : i32
  %6:3 = rtg.validate %0, %1, "id1" (%2, %3 else %4, %5 : i32, i32) : !rtgtest.ireg -> i32
  rtgtest.lui %0, %6#0 : i32
  rtgtest.lui %0, %6#1 : i32
  rtgtest.lui %0, %6#2 : i32
}

//--- test7.mlir
// RUN: circt-opt %t/test7.mlir --rtg-embed-validation-values=filename=%S/validation-values-3.txt | FileCheck %s --check-prefix=CHECK-TEST7

// CHECK-TEST7-LABEL: rtg.test @embed_value
rtg.test @embed_value() {
  // CHECK-TEST7-NEXT: [[REG:%.+]] = rtg.constant #rtgtest.t0
  // CHECK-TEST7-NEXT: [[IMM:%.+]] = rtg.constant 4096 : i32
  // CHECK-TEST7-NEXT: rtgtest.lui [[REG]], [[IMM]] : i32
  %reg = rtg.constant #rtgtest.t0
  %imm = rtg.constant 0x1000 : i32
  %exp_val = rtg.validate %reg, %imm, "id1" : !rtgtest.ireg -> i32
  rtgtest.lui %reg, %exp_val : i32
}
