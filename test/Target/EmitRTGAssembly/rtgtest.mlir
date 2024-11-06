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
  // CHECK: .word 0x1C3400
  // CHECK-ALLOWED: instr_b i3, f1, v0
  %regi = rtgtest.ireg 3
  %regf = rtgtest.freg 5
  %regv = rtgtest.vreg 10
  rtgtest.instr_b %regi, %regf, %regv
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
