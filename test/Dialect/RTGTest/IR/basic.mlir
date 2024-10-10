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
%reg5 = rtgtest.reg_a 5
%reg2 = rtgtest.reg_a 2
rtgtest.instr_b %reg5, %reg2


func.func @checkOnContext(
    %arg1 : !rtg.context_resource_set<!rtgtest.bogus_reg>
  ) {
  %la = rtg.label.decl "a" -> i32
  %lb = rtg.label.decl "b" -> i32
  %lc = rtg.label.decl "c" -> i32
  %ld = rtg.label.decl "d" -> i32
  %le = rtg.label.decl "e" -> i32
  rtg.label %la : i32
  %arg2 = rtg.select_random_resource %arg1 : !rtg.context_resource_set<!rtgtest.bogus_reg>
  rtg.on_context %arg1 : !rtg.context_resource_set<!rtgtest.bogus_reg> {
    rtg.label %lb : i32
    rtg.on_context %arg2 : !rtg.context_resource_set<!rtgtest.bogus_reg> {
      rtg.label %lc : i32
    }
    rtg.label %ld : i32
  }
  %arg3 = rtg.set_difference_resource %arg1, %arg2 : !rtg.context_resource_set<!rtgtest.bogus_reg>
  rtg.on_context %arg3 : !rtg.context_resource_set<!rtgtest.bogus_reg> {
    rtg.label %le : i32
  }
  return
}
