// RUN: circt-translate --emit-assembly %s | FileCheck %s --check-prefix=CHECK-ALLOWED
// RUN: circt-translate --emit-assembly --emit-assembly-binary-instr="rtgtest.instr_a,rtgtest.instr_b" %s | FileCheck %s 
// RUN: circt-opt --rtg-context %s | FileCheck %s  --check-prefix=CHECK-TRANS

rtg.sequence {
  // CHECK: .word 0x710000000400000004 
  // CHECK-ALLOWED: instr_a 4, 4
  %c = arith.constant 4 : i32
  rtgtest.instr_a %c, %c
} -> !rtg.sequence

rtg.sequence {
  %0 = rtg.label.decl "label0" -> i32
  // CHECK: label0:
  // CHECK-ALLOWED: label0:
  rtg.label %0 : i32
  // CHECK: .global label0
  // CHECK-ALLOWED: .global label0
  // CHECK: label0:
  // CHECK-ALLOWED: label0:
  rtg.label global %0 : i32
  // CHECK: .word 0xE00A0400 
  // CHECK-ALLOWED: instr_b x5, x2
  %reg5 = rtgtest.reg_a 5
  %reg2 = rtgtest.reg_a 2
  rtgtest.instr_b %reg5, %reg2
} -> !rtg.sequence

// CHECK-TRANS: [1, 2, 3, 4]
rtg.rendered_context [1, 2]
{
  %c = arith.constant 4 : i32
  rtgtest.instr_a %c, %c
},
{
  %c = arith.constant 5 : i32
  rtgtest.instr_a %c, %c
}


rtg.rendered_context [1,2,3]
{
  %0 = rtg.label.decl "label0" -> i32
  rtg.label %0 : i32
},
{
  %0 = rtg.label.decl "label1" -> i32
  rtg.label %0 : i32
},
{
  %0 = rtg.label.decl "label2" -> i32
  rtg.label %0 : i32
}

rtg.rendered_context [2,3,4]
{
  %0 = rtg.label.decl "labela3" -> i32
  rtg.label %0 : i32
},
{
  %0 = rtg.label.decl "labela4" -> i32
  rtg.label %0 : i32
},
{
  %0 = rtg.label.decl "labela5" -> i32
  rtg.label %0 : i32
}
