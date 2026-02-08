// RUN: circt-opt --rtg-emit-isa-assembly %s 2>&1 >/dev/null | FileCheck %s --match-full-lines --strict-whitespace

emit.file "" {
  %str_begin = rtg.constant "Begin of test0" : !rtg.string
  // CHECK:    # Begin of test0
  rtg.comment %str_begin
  %rd = rtg.constant #rtgtest.ra
  %rs = rtg.constant #rtgtest.s0
  %label = rtg.constant #rtg.isa.label<"label_name">

  // CHECK-NEXT:    la ra, label_name
  rtgtest.rv32i.la %rd, %label : !rtg.isa.label
  // CHECK-NEXT:    beq ra, s0, label_name
  rtgtest.rv32i.beq %rd, %rs, %label : !rtg.isa.label
  // CHECK-NEXT:label_name:
  rtg.label local %label
  // CHECK-NEXT:.extern label_name
  rtg.label external %label
  // CHECK-NEXT:.global label_name
  // CHECK-NEXT:label_name:
  rtg.label global %label
  // CHECK-NEXT:    bne ra, s0, label_name
  rtgtest.rv32i.bne %rd, %rs, %label : !rtg.isa.label
  // CHECK-NEXT:    blt ra, s0, label_name
  rtgtest.rv32i.blt %rd, %rs, %label : !rtg.isa.label
  // CHECK-NEXT:    bge ra, s0, label_name
  rtgtest.rv32i.bge %rd, %rs, %label : !rtg.isa.label
  // CHECK-NEXT:    bltu ra, s0, label_name
  rtgtest.rv32i.bltu %rd, %rs, %label : !rtg.isa.label
  // CHECK-NEXT:    bgeu ra, s0, label_name
  rtgtest.rv32i.bgeu %rd, %rs, %label : !rtg.isa.label
  // CHECK-NEXT:    lui ra, label_name
  rtgtest.rv32i.lui %rd, %label : !rtg.isa.label
  // CHECK-NEXT:    auipc ra, label_name
  rtgtest.rv32i.auipc %rd, %label : !rtg.isa.label
  // CHECK-NEXT:    jal ra, label_name
  rtgtest.rv32i.jal %rd, %label : !rtg.isa.label
}
