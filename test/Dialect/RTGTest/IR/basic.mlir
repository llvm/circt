// RUN: circt-opt %s | FileCheck %s

// CHECK: [[IMM1:%.+]] = arith.constant 4 : i32
// CHECK: [[IMM2:%.+]] = arith.constant 8 : i32
// CHECK: rtgtest.instr_a [[IMM1]], [[IMM2]]
%imm1 = arith.constant 4 : i32
%imm2 = arith.constant 8 : i32
rtgtest.instr_a %imm1, %imm2



func.func @checkOnContext(
    %arg1 : !rtg.context_resource_set<!rtgtest.bogus_reg>, 
    %arg2: !rtg.context_resource_set<!rtgtest.bogus_reg>
  ) {
  rtg.label "a"
  rtg.on_context %arg1 : !rtg.context_resource_set<!rtgtest.bogus_reg> {
    rtg.label "b"
    rtg.on_context %arg2 : !rtg.context_resource_set<!rtgtest.bogus_reg> {
      rtg.label "c"
    }
    rtg.label "d"
  }
  rtg.on_context %arg2 : !rtg.context_resource_set<!rtgtest.bogus_reg> {
    rtg.label "e"
  }
  return
}
