// RUN: circt-translate --emit-assembly %s --verify-diagnostics --split-input-file

rtg.snippet {
  // expected-error @below {{binary representation cannot be computed for labels}}
  %0 = rtg.label.decl "label0" -> i32
  rtg.label %0 : i32
  rtgtest.instr_a %0, %0
}
