// RUN: circt-opt --rtg-emit-isa-assembly=unsupported-instructions=rtgtest.memory_instr %s --split-input-file --verify-diagnostics

emit.file "-" {
  %rd = rtg.constant #rtgtest.ra
  %label = rtg.constant #rtg.isa.label<"label_name">

  // expected-error @below {{labels cannot be emitted as binary}}
  rtgtest.memory_instr %rd, %label : !rtg.isa.label
}

// -----

emit.file "-" {
  // expected-error @below {{implicit constraint not materialized}}
  rtgtest.implicit_constraint_op implicit_constraint
}
