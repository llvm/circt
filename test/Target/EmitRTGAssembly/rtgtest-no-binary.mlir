// RUN: circt-translate --emit-assembly %s | FileCheck %s

/// Labels can probably be dealt with in binary emission is you know the 
// addressing style of the op.  This might require some effort...

rtg.sequence {
  // CHECK: label0:
  %0 = rtg.label.decl "label0" -> i32
  rtg.label %0 : i32
  // CHECK: instr_a label0, label0
  rtgtest.instr_a %0, %0
} -> !rtg.sequence

