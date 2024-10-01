// RUN: circt-translate --emit-assembly %s | FileCheck %s
// RUN: circt-translate --emit-assembly --emit-assembly-allowed-instr="rtgtest.instr_a" %s | FileCheck %s --check-prefix=CHECK-ALLOWED

rtg.snippet {
  // CHECK: .word 0x710000000400000004 
  // CHECK-ALLOWED: instr_a 4, 4
  %c = arith.constant 4 : i32
  rtgtest.instr_a %c, %c
}

rtg.rendered_context [1, 2]
{
  %c = arith.constant 4 : i32
  rtgtest.instr_a %c, %c
},
{
  %c = arith.constant 5 : i32
  rtgtest.instr_a %c, %c
}
