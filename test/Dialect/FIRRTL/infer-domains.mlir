// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-infer-domains{mode=infer-all}))' %s | FileCheck %s

// Legal domain usage - no crossing.
// CHECK-LABEL: firrtl.circuit "LegalDomains"
firrtl.circuit "LegalDomains" {
  firrtl.domain @ClockDomain
  firrtl.module @LegalDomains(
    in  %A: !firrtl.domain of @ClockDomain,
    in  %a: !firrtl.uint<1> domains [%A],
    out %b: !firrtl.uint<1> domains [%A]
  ) {
    // Connecting within the same domain is legal.
    firrtl.matchingconnect %b, %a : !firrtl.uint<1>
  }
}

// Domain inference through connections.
// CHECK-LABEL: firrtl.circuit "DomainInference"
firrtl.circuit "DomainInference" {
  firrtl.domain @ClockDomain
  firrtl.module @DomainInference(
    in %A: !firrtl.domain of @ClockDomain,
    in %a: !firrtl.uint<1> domains [%A],
    // CHECK: out %c: !firrtl.uint<1> domains [%A]
    out %c: !firrtl.uint<1>
  ) {
    %b = firrtl.wire : !firrtl.uint<1>  // No explicit domain

    // This should infer that %b is in domain %A.
    firrtl.matchingconnect %b, %a : !firrtl.uint<1>

    // This should be legal since %b is now inferred to be in domain %A.
    firrtl.matchingconnect %c, %b : !firrtl.uint<1>
  }
}

// Unsafe domain cast
// CHECK-LABEL: firrtl.circuit "UnsafeDomainCast"
firrtl.circuit "UnsafeDomainCast" {
  firrtl.domain @ClockDomain
  firrtl.module @UnsafeDomainCast(
    in %A: !firrtl.domain of @ClockDomain,
    in %B: !firrtl.domain of @ClockDomain,
    in %a: !firrtl.uint<1> domains [%A],
    out %c: !firrtl.uint<1> domains [%B]
  ) {
    // Unsafe cast from domain A to domain B.
    %b = firrtl.unsafe_domain_cast %a domains %B : !firrtl.uint<1>

    // This should be legal since we explicitly cast.
    firrtl.matchingconnect %c, %b : !firrtl.uint<1>
  }
}

// Domain sequence matching.
// CHECK-LABEL: firrtl.circuit "LegalSequences"
firrtl.circuit "LegalSequences" {
  firrtl.domain @ClockDomain
  firrtl.domain @PowerDomain
  firrtl.module @LegalSequences(
    in  %C: !firrtl.domain of @ClockDomain,
    in  %P: !firrtl.domain of @PowerDomain,
    in  %a: !firrtl.uint<1> domains [%C, %P],
    out %b: !firrtl.uint<1> domains [%C, %P]
  ) {
    firrtl.matchingconnect %b, %a : !firrtl.uint<1>
  }
}

// Domain sequence order equivalence - should be legal
// CHECK-LABEL: SequenceOrderEquivalence
firrtl.circuit "SequenceOrderEquivalence" {
  firrtl.domain @ClockDomain
  firrtl.domain @PowerDomain
  firrtl.module @SequenceOrderEquivalence(
    in %A: !firrtl.domain of @ClockDomain,
    in %B: !firrtl.domain of @PowerDomain,
    in %a: !firrtl.uint<1> domains [%A, %B],
    out %b: !firrtl.uint<1> domains [%B, %A]
  ) {
    // This should be legal since domain order doesn't matter in canonical representation
    firrtl.matchingconnect %b, %a : !firrtl.uint<1>
  }
}

// Domain sequence inference
// CHECK-LABEL: SequenceInference
firrtl.circuit "SequenceInference" {
  firrtl.domain @ClockDomain
  firrtl.domain @PowerDomain
  firrtl.module @SequenceInference(
    in %A: !firrtl.domain of @ClockDomain,
    in %B: !firrtl.domain of @PowerDomain,
    in %a: !firrtl.uint<1> domains [%A, %B],
    out %d: !firrtl.uint<1>
  ) {
    %c = firrtl.wire : !firrtl.uint<1>

    // %c should infer domain sequence [%A, %B]
    firrtl.matchingconnect %c, %a : !firrtl.uint<1>

    // This should be legal since %c has inferred [%A, %B]
    firrtl.matchingconnect %d, %c : !firrtl.uint<1>
  }
}

// Domain duplicate equivalence - should be legal.
// CHECK-LABEL: DuplicateDomainEquivalence
firrtl.circuit "DuplicateDomainEquivalence" {
  firrtl.domain @ClockDomain
  firrtl.module @DuplicateDomainEquivalence(
    in %A: !firrtl.domain of @ClockDomain,
    in %a: !firrtl.uint<1> domains [%A, %A],
    out %b: !firrtl.uint<1> domains [%A]
  ) {
    // This should be legal since duplicate domains are canonicalized.
    firrtl.matchingconnect %b, %a : !firrtl.uint<1>
  }
}

// Unsafe domain cast with sequences
// CHECK-LABEL: UnsafeSequenceCast
firrtl.circuit "UnsafeSequenceCast" {
  firrtl.domain @ClockDomain
  firrtl.domain @PowerDomain

  firrtl.module @UnsafeSequenceCast(
    in %C1: !firrtl.domain of @ClockDomain,
    in %C2: !firrtl.domain of @ClockDomain,
    in %P1: !firrtl.domain of @PowerDomain,
    in  %i: !firrtl.uint<1> domains [%C1, %P1],
    out %o: !firrtl.uint<1> domains [%C2, %P1]
  ) {
    %0 = firrtl.unsafe_domain_cast %i domains %C2 : !firrtl.uint<1>
    firrtl.matchingconnect %o, %0 : !firrtl.uint<1>
  }
}

//  Different port types domain inference.
// CHECK-LABEL: DifferentPortTypes
firrtl.circuit "DifferentPortTypes" {
  firrtl.domain @ClockDomain
  firrtl.module @DifferentPortTypes(
    in %A: !firrtl.domain of @ClockDomain,
    in %uint_input: !firrtl.uint<8> domains [%A],
    in %sint_input: !firrtl.sint<4> domains [%A],
    out %uint_output: !firrtl.uint<8>,
    out %sint_output: !firrtl.sint<4>
  ) {
    firrtl.matchingconnect %uint_output, %uint_input : !firrtl.uint<8>
    firrtl.matchingconnect %sint_output, %sint_input : !firrtl.sint<4>
  }
}

// Domain inference through wires.
// CHECK-LABEL: DomainInferenceThroughWires
firrtl.circuit "DomainInferenceThroughWires" {
  firrtl.domain @ClockDomain
  firrtl.module @DomainInferenceThroughWires(
    in %A: !firrtl.domain of @ClockDomain,
    in %input: !firrtl.uint<1> domains [%A],
    // CHECK: out %output: !firrtl.uint<1> domains [%A]
    out %output: !firrtl.uint<1>
  ) {
    %wire1 = firrtl.wire : !firrtl.uint<1>
    %wire2 = firrtl.wire : !firrtl.uint<1>

    firrtl.matchingconnect %wire1, %input : !firrtl.uint<1>
    firrtl.matchingconnect %wire2, %wire1 : !firrtl.uint<1>
    firrtl.matchingconnect %output, %wire2 : !firrtl.uint<1>
  }
}

// Export: add output domain port for domain created internally.
// CHECK-LABEL: ExportDomain
firrtl.circuit "ExportDomain" {
  firrtl.domain @ClockDomain

  firrtl.extmodule @Foo(
    out A: !firrtl.domain of @ClockDomain,
    out o: !firrtl.uint<1> domains [A]
  )

  firrtl.module @ExportDomain(
    // CHECK: out %ClockDomain: !firrtl.domain of @ClockDomain
    // CHECK: out %o: !firrtl.uint<1> domains [%ClockDomain]
    out %o: !firrtl.uint<1>
  ) {
    %foo_A, %foo_o = firrtl.instance foo @Foo(
      out A: !firrtl.domain of @ClockDomain,
      out o: !firrtl.uint<1> domains [A]
    )
    firrtl.matchingconnect %o, %foo_o : !firrtl.uint<1>
    // CHECK: firrtl.domain.define %ClockDomain, %foo_A
  }
}

// Export: Reuse already-exported domain.
// CHECK-LABEL: ReuseExportedDomain
firrtl.circuit "ReuseExportedDomain" {
    firrtl.domain @ClockDomain

  firrtl.extmodule @Foo(
    out A: !firrtl.domain of @ClockDomain,
    out o: !firrtl.uint<1> domains [A]
  )

  firrtl.module @ReuseExportedDomain(
    out %A: !firrtl.domain of @ClockDomain,
    // CHECK: out %o: !firrtl.uint<1> domains [%A]
    out %o: !firrtl.uint<1>
  ) {
    %foo_A, %foo_o = firrtl.instance foo @Foo(
      out A: !firrtl.domain of @ClockDomain,
      out o: !firrtl.uint<1> domains [A]
    )
    firrtl.matchingconnect %o, %foo_o : !firrtl.uint<1>
    firrtl.domain.define %A, %foo_A
  }
}

// CHECK-LABEL: RegisterInference
firrtl.circuit "RegisterInference" {
  firrtl.domain @ClockDomain
  firrtl.module @RegisterInference(
    in  %A: !firrtl.domain of @ClockDomain,
    in  %clock: !firrtl.clock domains [%A],
    // CHECK: in %d: !firrtl.uint<1> domains [%A]
    in  %d: !firrtl.uint<1>,
    // CHECK: out %q: !firrtl.uint<1> domains [%A]
    out %q: !firrtl.uint<1>
  ) {
    %r = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<1>
    firrtl.matchingconnect %r, %d : !firrtl.uint<1>
    firrtl.matchingconnect %q, %r : !firrtl.uint<1>
  }
}

// CHECK-LABEL: InstanceUpdate
firrtl.circuit "InstanceUpdate" {
  firrtl.domain @ClockDomain

  firrtl.module @Foo(in %i : !firrtl.uint<1>) {}

  // CHECK: firrtl.module @InstanceUpdate(in %ClockDomain: !firrtl.domain of @ClockDomain, in %i: !firrtl.uint<1> domains [%ClockDomain]) {
  // CHECK:   %foo_ClockDomain, %foo_i = firrtl.instance foo @Foo(in ClockDomain: !firrtl.domain of @ClockDomain, in i: !firrtl.uint<1> domains [ClockDomain])
  // CHECK:   firrtl.domain.define %foo_ClockDomain, %ClockDomain
  // CHECK:   firrtl.connect %foo_i, %i : !firrtl.uint<1>
  // CHECK: }
  firrtl.module @InstanceUpdate(in %i : !firrtl.uint<1>) {
    %foo_i = firrtl.instance foo @Foo(in i: !firrtl.uint<1>)
    firrtl.connect %foo_i, %i : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// CHECK-LABEL: InstanceChoiceUpdate
firrtl.circuit "InstanceChoiceUpdate" {
  firrtl.domain @ClockDomain

  firrtl.option @Option {
    firrtl.option_case @X
    firrtl.option_case @Y
  }

  firrtl.module @Foo(in %i : !firrtl.uint<1>) {}
  firrtl.module @Bar(in %i : !firrtl.uint<1>) {}
  firrtl.module @Baz(in %i : !firrtl.uint<1>) {}

  // CHECK: firrtl.module @InstanceChoiceUpdate(in %ClockDomain: !firrtl.domain of @ClockDomain, in %i: !firrtl.uint<1> domains [%ClockDomain]) {
  // CHECK:   %inst_ClockDomain, %inst_i = firrtl.instance_choice inst @Foo alternatives @Option { @X -> @Bar, @Y -> @Baz } (in ClockDomain: !firrtl.domain of @ClockDomain, in i: !firrtl.uint<1> domains [ClockDomain])
  // CHECK:   firrtl.domain.define %inst_ClockDomain, %ClockDomain
  // CHECK:   firrtl.connect %inst_i, %i : !firrtl.uint<1>
  // CHECK: }
  firrtl.module @InstanceChoiceUpdate(in %i : !firrtl.uint<1>) {
    %inst_i = firrtl.instance_choice inst @Foo alternatives @Option { @X -> @Bar, @Y -> @Baz } (in i : !firrtl.uint<1>)
    firrtl.connect %inst_i, %i : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// CHECK-LABEL: ConstantInMultipleDomains
firrtl.circuit "ConstantInMultipleDomains" {
  firrtl.domain @ClockDomain

  firrtl.extmodule @Foo(in A: !firrtl.domain of @ClockDomain, in i: !firrtl.uint<1> domains [A])

  firrtl.module @ConstantInMultipleDomains(in %A: !firrtl.domain of @ClockDomain, in %B: !firrtl.domain of @ClockDomain) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %x_A, %x_i = firrtl.instance x @Foo(in A: !firrtl.domain of @ClockDomain, in i: !firrtl.uint<1> domains [A])
    firrtl.domain.define %x_A, %A
    firrtl.matchingconnect %x_i, %c0_ui1 : !firrtl.uint<1>
    
    %y_A, %y_i = firrtl.instance y @Foo(in A: !firrtl.domain of @ClockDomain, in i: !firrtl.uint<1> domains [A])
    firrtl.domain.define %y_A, %B
    firrtl.matchingconnect %y_i, %c0_ui1 : !firrtl.uint<1>
  }
}

firrtl.circuit "Top" {
  firrtl.domain @ClockDomain
  firrtl.extmodule @Foo(
    in ClockDomain : !firrtl.domain of @ClockDomain,
    in i: !firrtl.uint<1> domains [ClockDomain],
    out o : !firrtl.uint<1> domains [ClockDomain]
  )

  firrtl.module @Top(in %ClockDomain : !firrtl.domain of @ClockDomain ) {
    %foo1_ClockDomain, %foo1_i, %foo1_o = firrtl.instance foo1 @Foo(
      in ClockDomain : !firrtl.domain of @ClockDomain,
      in i: !firrtl.uint<1> domains [ClockDomain],
      out o : !firrtl.uint<1> domains [ClockDomain]
    )

    %foo2_ClockDomain, %foo2_i, %foo2_o = firrtl.instance foo2 @Foo(
      in ClockDomain : !firrtl.domain of @ClockDomain,
      in i: !firrtl.uint<1> domains [ClockDomain],
      out o : !firrtl.uint<1> domains [ClockDomain]
    )

    firrtl.domain.define %foo1_ClockDomain, %ClockDomain
    firrtl.matchingconnect %foo2_i, %foo1_o : !firrtl.uint<1>
    firrtl.matchingconnect %foo1_i, %foo2_o : !firrtl.uint<1>
  }
}

// Ensure that unique domain names are created.
// CHECK-LABEL: firrtl.circuit "NoDuplicateNames"
firrtl.circuit "NoDuplicateNames" {
  firrtl.domain @ClockDomain
  firrtl.extmodule private @Bar(
    in A: !firrtl.domain of @ClockDomain,
    in B: !firrtl.domain of @ClockDomain,
    in a: !firrtl.uint<1> domains [A],
    out b: !firrtl.uint<1> domains [B]
  )
  // CHECK:      firrtl.module @NoDuplicateNames
  // CHECK-SAME:   in %ClockDomain:
  // CHECK-SAME:   in %{{ClockDomain[A-Za-z0-9_]+}}:
  // CHECK-NOT:    portNames
  firrtl.module @NoDuplicateNames(
    in %a: !firrtl.uint<1>,
    out %b: !firrtl.uint<1>
  ) {
    %bar_A, %bar_B, %bar_a, %bar_b = firrtl.instance bar @Bar(
      in A: !firrtl.domain of @ClockDomain,
      in B: !firrtl.domain of @ClockDomain,
      in a: !firrtl.uint<1> domains [A],
      out b: !firrtl.uint<1> domains [B]
    )
    firrtl.matchingconnect %bar_a, %a : !firrtl.uint<1>
    firrtl.matchingconnect %b, %bar_b : !firrtl.uint<1>
  }
}
