// RUN: circt-opt %s | FileCheck %s

// CHECK: [[IMM1:%.+]] = arith.constant 4 : i32
// CHECK: [[IMM2:%.+]] = arith.constant 8 : i32
// CHECK: rtgtest.instr_a [[IMM1]], [[IMM2]]
%imm1 = arith.constant 4 : i32
%imm2 = arith.constant 8 : i32
rtgtest.instr_a %imm1, %imm2



func.func @checkOnContext(
    %arg1 : !rtg.context_resource_set<!rtgtest.bogus_reg>
  ) {
  rtg.label "a"
  %arg2 = rtg.select_random_context_resource %arg1 : !rtg.context_resource_set<!rtgtest.bogus_reg>
  rtg.on_context %arg1 : !rtg.context_resource_set<!rtgtest.bogus_reg> {
    rtg.label "b"
    rtg.on_context %arg2 : !rtg.context_resource_set<!rtgtest.bogus_reg> {
      rtg.label "c"
    }
    rtg.label "d"
  }
  %arg3 = rtg.set_difference_resource %arg1, %arg2 : !rtg.context_resource_set<!rtgtest.bogus_reg>
  rtg.on_context %arg3 : !rtg.context_resource_set<!rtgtest.bogus_reg> {
    rtg.label "e"
  }
  return
}
