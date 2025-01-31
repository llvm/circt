// RUN: circt-opt --rtg-emit-isa-assembly=unsupported-instructions=rtgtest.rv32i.beq %s --split-input-file --verify-diagnostics

rtg.test @test0 : !rtg.dict<> {
  %rd = rtg.fixed_reg #rtgtest.ra
  %rs = rtg.fixed_reg #rtgtest.s0
  %label = rtg.label_decl "label_name"

  // expected-error @below {{labels cannot be emitted as binary}}
  rtgtest.rv32i.beq %rd, %rs, %label : !rtg.label
}

// -----

rtg.test @test0 : !rtg.dict<> {
  %0 = index.constant 0
  // expected-error @below {{label arguments must be elaborated before emission}}
  %label = rtg.label_decl "label_name_{{0}}", %0
}
