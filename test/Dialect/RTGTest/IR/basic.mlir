// RUN: circt-opt %s | FileCheck %s

// CHECK: [[IMM1:%.+]] = arith.constant 4 : i32
// CHECK: [[IMM2:%.+]] = arith.constant 8 : i32
// CHECK: rtgtest.instr_a [[IMM1]], [[IMM2]]
%imm1 = arith.constant 4 : i32
%imm2 = arith.constant 8 : i32
rtgtest.instr_a %imm1, %imm2

// CHECK: [[REG0:%.+]] = rtgtest.reg_a 5
// CHECK: [[REG1:%.+]] = rtgtest.reg_a 2
// CHECK: rtgtest.instr_b [[REG0]], [[REG1]]
%reg5 = rtgtest.ireg 5
%reg2 = rtgtest.freg 2
%reg3 = rtgtest.vreg 3
rtgtest.instr_b %reg5, %reg2, %reg3


func.func @checkOnContext(
    %arg1 : !rtg.set<!rtgtest.dummyCPUs>
  ) {
  %la = rtg.label.decl "a" -> i32
  %lb = rtg.label.decl "b" -> i32
  %lc = rtg.label.decl "c" -> i32
  %ld = rtg.label.decl "d" -> i32
  %le = rtg.label.decl "e" -> i32
  rtg.label %la : i32
  %arg2 = rtg.set_select_random %arg1 : !rtg.set<!rtgtest.dummyCPUs>
  rtg.on_context %arg1 : !rtg.set<!rtgtest.dummyCPUs> {
    rtg.label %lb : i32
    rtg.on_context %arg2 : !rtg.set<!rtgtest.dummyCPUs> {
      rtg.label %lc : i32
    }
    rtg.label %ld : i32
  }
  %arg3 = rtg.set_difference %arg1, %arg2 : !rtg.set<!rtgtest.bogus_reg>
  rtg.on_context %arg3 : !rtg.set<!rtgtest.bogus_reg> {
    rtg.label %le : i32
  }
  return
}
