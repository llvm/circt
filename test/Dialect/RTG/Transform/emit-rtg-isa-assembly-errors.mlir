// RUN: circt-opt --rtg-emit-isa-assembly=unsupported-instructions=rtgtest.rv32i.beq %s --split-input-file --verify-diagnostics

emit.file "-" {
  %rd = rtg.constant #rtgtest.ra
  %rs = rtg.constant #rtgtest.s0
  %label = rtg.label_decl "label_name"

  // expected-error @below {{labels cannot be emitted as binary}}
  rtgtest.rv32i.beq %rd, %rs, %label : !rtg.isa.label
}

// -----

emit.file "-" {
  %0 = index.constant 0
  // expected-error @below {{label arguments must be elaborated before emission}}
  %label = rtg.label_decl "label_name_{{0}}", %0
}

// -----

emit.file "-" {
  // expected-error @below {{implicit constraint not materialized}}
  rtgtest.implicit_constraint_op implicit_constraint
}
