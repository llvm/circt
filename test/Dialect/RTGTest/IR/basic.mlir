// RUN: circt-opt %s | FileCheck %s

// CHECK: [[IMM1:%.+]] = arith.constant 4 : i32
// CHECK: [[IMM2:%.+]] = arith.constant 8 : i32
// CHECK: rtgtest.instr_a [[IMM1]], [[IMM2]]
%imm1 = arith.constant 4 : i32
%imm2 = arith.constant 8 : i32
rtgtest.instr_a %imm1, %imm2
