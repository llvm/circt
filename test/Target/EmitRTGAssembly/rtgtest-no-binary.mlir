// RUN: circt-translate --emit-assembly --emit-assembly-allowed-instr="rtgtest.instr_a,rtgtest.instr_b" %s | FileCheck %s

rtg.snippet {
  // CHECK: label0:
  %0 = rtg.label.decl "label0" -> i32
  rtg.label %0 : i32
  // CHECK: instr_a label0, label0
  rtgtest.instr_a %0, %0
}
