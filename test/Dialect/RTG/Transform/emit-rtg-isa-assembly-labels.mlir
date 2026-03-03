// RUN: circt-opt --rtg-emit-isa-assembly %s 2>&1 >/dev/null | FileCheck %s --match-full-lines --strict-whitespace

emit.file "" {
  %str_begin = rtg.constant "Begin of test0" : !rtg.string
  // CHECK:    # Begin of test0
  rtg.comment %str_begin
  %rd = rtg.constant #rtgtest.ra
  %rs = rtg.constant #rtgtest.s0
  %label = rtg.constant #rtg.isa.label<"label_name">

  // CHECK-NEXT:    memory_instr ra, label_name
  rtgtest.memory_instr %rd, %label : !rtg.isa.label
  // CHECK-NEXT:label_name:
  rtg.label local %label
  // CHECK-NEXT:.extern label_name
  rtg.label external %label
  // CHECK-NEXT:.global label_name
  // CHECK-NEXT:label_name:
  rtg.label global %label
}
