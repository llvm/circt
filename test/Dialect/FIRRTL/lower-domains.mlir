// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-domains))' --split-input-file %s | FileCheck --implicit-check-not firrtl.domain --implicit-check-not unrealized_conversion_cast %s

// Deeply check the lowering of domains and how associations are lowered.  Later
// tests will ignore these deep checks.
firrtl.circuit "Foo" {
  // Two classes should be created:
  //   1. A first that just represents the domain.  This has information that a
  //      user needs to provide.
  //   2. A second that returns this user-provided information and includes the
  //      domain information.
  // CHECK-LABEL: firrtl.class @ClockDomain
  // CHECK-LABEL: firrtl.class @ClockDomain_out
  // CHECK-SAME:    in %domainInfo_in: !firrtl.class<@ClockDomain()>
  // CHECK-SAME:    out %domainInfo_out: !firrtl.class<@ClockDomain()>
  // CHECK-SAME:    in %associations_in: !firrtl.list<path>
  // CHECK-SAME:    out %associations_out: !firrtl.list<path>
  // CHECK-NEXT:    firrtl.propassign %domainInfo_out, %domainInfo_in
  // CHECK-NEXT:    firrtl.propassign %associations_out, %associations_in
  firrtl.domain @ClockDomain
  // CHECK-LABEL: firrtl.module @Foo(
  // CHECK-SAME:    in %A: !firrtl.class<@ClockDomain()>
  // CHECK-SAME:    out %A_out: !firrtl.class<@ClockDomain_out(
  // CHECK-SAME:    in %a: !firrtl.uint<1> [{class = "circt.tracker", id = distinct[0]<>}]
  firrtl.module @Foo(
    in %A: !firrtl.domain of @ClockDomain,
    in %a: !firrtl.uint<1> domains [%A]
    // CHECK:      %A_object = firrtl.object @ClockDomain
    // CHECK-NEXT: %[[domainInfo_in:.+]] = firrtl.object.subfield %A_object[domainInfo_in]
    // CHECK-NEXT: firrtl.propassign %[[domainInfo_in]], %A :
    // CHECK-NEXT: %[[associations_in:.+]] = firrtl.object.subfield %A_object[associations_in]
    // CHECK-NEXT: %[[path:.+]] = firrtl.path reference distinct[0]<>
    // CHECK-NEXT: %[[list:.+]] = firrtl.list.create %[[path]] :
    // CHECK-NEXT: firrtl.propassign %[[associations_in]], %[[list]]
    // CHECK-NEXT: firrtl.propassign %A_out, %A_object :
  ) {}
}

// -----

// Check that multiple domain associations work.
firrtl.circuit "Foo" {
  // CHECK-LABEL: firrtl.class @ClockDomain(
  // CHECK-LABEL: firrtl.class @ClockDomain_out(
  // CHECK-SAME:    in %domainInfo_in: !firrtl.class<@ClockDomain()>
  // CHECK-SAME:    out %domainInfo_out: !firrtl.class<@ClockDomain()>
  // CHECK-SAME:    in %associations_in: !firrtl.list<path>
  // CHECK-SAME:    out %associations_out: !firrtl.list<path>
  firrtl.domain @ClockDomain
  // CHECK-LABEL: firrtl.class @ResetDomain(
  // CHECK-LABEL: firrtl.class @ResetDomain_out(
  // CHECK-SAME:    in %domainInfo_in: !firrtl.class<@ResetDomain()>
  // CHECK-SAME:    out %domainInfo_out: !firrtl.class<@ResetDomain()>
  // CHECK-SAME:    in %associations_in: !firrtl.list<path>
  // CHECK-SAME:    out %associations_out: !firrtl.list<path>
  firrtl.domain @ResetDomain
  // CHECK-LABEL: firrtl.module @Foo(
  // CHECK-SAME:    in %A: !firrtl.class<@ClockDomain()>
  // CHECK-SAME:    out %A_out: !firrtl.class<@ClockDomain_out(
  // CHECK-SAME:    in %B: !firrtl.class<@ResetDomain()>
  // CHECK-SAME:    out %B_out: !firrtl.class<@ResetDomain_out(
  // CHECK-SAME:    in %a: !firrtl.uint<1> [{class = "circt.tracker", id = distinct[0]<>}]
  firrtl.module @Foo(
    in %A: !firrtl.domain of @ClockDomain,
    in %B: !firrtl.domain of @ResetDomain,
    in %a: !firrtl.uint<1> domains [%A, %B]
  ) {
    // CHECK:      %A_object = firrtl.object @ClockDomain
    // CHECK:      %[[clock_associations_in:.+]] = firrtl.object.subfield %A_object[associations_in]
    // CHECK-NEXT: %[[path:.+]] = firrtl.path reference distinct[0]<>
    // CHECK-NEXT: %[[list:.+]] = firrtl.list.create %[[path]] :
    // CHECK-NEXT: firrtl.propassign %[[clock_associations_in]], %[[list]]
    //
    // CHECK:      %B_object = firrtl.object @ResetDomain
    // CHECK:      %[[reset_associations_in:.+]] = firrtl.object.subfield %B_object[associations_in]
    // CHECK-NEXT: %[[path:.+]] = firrtl.path reference distinct[0]<>
    // CHECK-NEXT: %[[list:.+]] = firrtl.list.create %[[path]] :
    // CHECK-NEXT: firrtl.propassign %[[reset_associations_in]], %[[list]]
  }
}

// -----

// Spot check that insertions and deletions of ports involving prime numbers of
// ports and different positions for domain ports works.  This test is
// intentionally NOT testing everything here due to the complexity of
// maintaining this test.
firrtl.circuit "Foo" {
  firrtl.domain @ClockDomain
  // CHECK-LABEL: firrtl.module @Foo
  // CHECK-SAME:  in %A:
  // CHECK-SAME:  out %A_out:
  // CHECK-SAME:  in %B:
  // CHECK-SAME:  out %B_out:
  // CHECK-SAME:  in %a0:
  // CHECK-SAME:  in %a1:
  // CHECK-SAME:  in %a2:
  // CHECK-SAME:  in %b0:
  // CHECK-SAME:  in %b1:
  // CHECK-SAME:  in %b2:
  // CHECK-SAME:  in %b3:
  // CHECK-SAME:  in %b4:
  firrtl.module @Foo(
    in %A: !firrtl.domain of @ClockDomain,
    in %B: !firrtl.domain of @ClockDomain,
    in %a0: !firrtl.uint<1> domains [%A],
    in %a1: !firrtl.uint<1> domains [%A],
    in %a2: !firrtl.uint<1> domains [%A],
    in %b0: !firrtl.uint<1> domains [%B],
    in %b1: !firrtl.uint<1> domains [%B],
    in %b2: !firrtl.uint<1> domains [%B],
    in %b3: !firrtl.uint<1> domains [%B],
    in %b4: !firrtl.uint<1> domains [%B]
  ) {
  }
  // CHECK-LABEL: firrtl.module @Bar
  // CHECK-SAME:  in %A:
  // CHECK-SAME:  out %A_out:
  // CHECK-SAME:  in %a0:
  // CHECK-SAME:  in %a1:
  // CHECK-SAME:  in %a2:
  // CHECK-SAME:  in %B:
  // CHECK-SAME:  out %B_out:
  // CHECK-SAME:  in %b0:
  // CHECK-SAME:  in %b1:
  // CHECK-SAME:  in %b2:
  // CHECK-SAME:  in %b3:
  // CHECK-SAME:  in %b4:
  firrtl.module @Bar(
    in %A: !firrtl.domain of @ClockDomain,
    in %a0: !firrtl.uint<1> domains [%A],
    in %a1: !firrtl.uint<1> domains [%A],
    in %a2: !firrtl.uint<1> domains [%A],
    in %B: !firrtl.domain of @ClockDomain,
    in %b0: !firrtl.uint<1> domains [%B],
    in %b1: !firrtl.uint<1> domains [%B],
    in %b2: !firrtl.uint<1> domains [%B],
    in %b3: !firrtl.uint<1> domains [%B],
    in %b4: !firrtl.uint<1> domains [%B]
  ) {
  }
}

// -----

// Test an "upwards" U-turn.
firrtl.circuit "Foo" {
  firrtl.domain @ClockDomain
  // CHECK-LABEL: firrtl.module @Foo(
  // CHECK-SAME:    in %A: !firrtl.class<@ClockDomain()>,
  // CHECK-SAME:    out %A_out: !firrtl.class<@ClockDomain_out(
  // CHECK-SAME:    out %B_out: !firrtl.class<@ClockDomain_out(
  // CHECK-SAME:    in %a: !firrtl.uint<1>
  // CHECK-SAME:      id = distinct[0]
  // CHECK-SAME:    out %b: !firrtl.uint<1>
  // CHECK-SAME:      id = distinct[1]
  firrtl.module @Foo(
    in %A: !firrtl.domain of @ClockDomain,
    out %B: !firrtl.domain of @ClockDomain,
    in %a: !firrtl.uint<1> domains [%A],
    out %b: !firrtl.uint<1> domains [%B]
  ) {
    // CHECK-NEXT: %A_object = firrtl.object @ClockDomain_out
    // CHECK-NEXT: %[[domainInfo_in:.+]] = firrtl.object.subfield %A_object[domainInfo_in]
    // CHECK-NEXT: firrtl.propassign %[[domainInfo_in]], %A :
    // CHECK-NEXT: %[[associations_in:.+]] = firrtl.object.subfield %A_object[associations_in]
    // CHECK-NEXT: %[[path:.+]] = firrtl.path reference distinct[0]<>
    // CHECK-NEXT: %[[list:.+]] = firrtl.list.create %[[path]] :
    // CHECK-NEXT: firrtl.propassign %[[associations_in]], %[[list]]
    // CHECK-NEXT: firrtl.propassign %A_out, %A_object :
    //
    // CHECK:      %B_object = firrtl.object @ClockDomain_out
    // CHECK-NEXT: %[[domainInfo_in:.+]] = firrtl.object.subfield %B_object[domainInfo_in]
    // CHECK-NEXT: %[[associations_in:.+]] = firrtl.object.subfield %B_object[associations_in]
    // CHECK-NEXT: %[[path:.+]] = firrtl.path reference distinct[1]<>
    // CHECK-NEXT: %[[list:.+]] = firrtl.list.create %[[path]] :
    // CHECK-NEXT: firrtl.propassign %[[associations_in]], %[[list]]
    // CHECK-NEXT: firrtl.propassign %B_out, %B_object :
    //
    // CHECK-NEXT: firrtl.propassign %[[domainInfo_in]], %A :
    firrtl.domain.define %B, %A
    // CHECK-NEXT: firrtl.matchingconnect %b, %a
    %0 = firrtl.unsafe_domain_cast %a domains %B : !firrtl.uint<1>
    firrtl.matchingconnect %b, %0 : !firrtl.uint<1>
  }
}

// -----

// Test an "upwards" U-turn involving wires.
firrtl.circuit "Foo" {
  firrtl.domain @ClockDomain
  // CHECK-LABEL: firrtl.module @Foo(
  // CHECK-SAME:    in %A: !firrtl.class<@ClockDomain()>,
  // CHECK-SAME:    out %A_out: !firrtl.class<@ClockDomain_out(
  // CHECK-SAME:    out %B_out: !firrtl.class<@ClockDomain_out(
  // CHECK-SAME:    in %a: !firrtl.uint<1>
  // CHECK-SAME:      id = distinct[0]
  // CHECK-SAME:    out %b: !firrtl.uint<1>
  // CHECK-SAME:      id = distinct[1]
  firrtl.module @Foo(
    in %A: !firrtl.domain of @ClockDomain,
    out %B: !firrtl.domain of @ClockDomain,
    in %a: !firrtl.uint<1> domains [%A],
    out %b: !firrtl.uint<1> domains [%B]
  ) {
    // CHECK-NEXT: %A_object = firrtl.object @ClockDomain_out
    // CHECK-NEXT: %[[domainInfo_in:.+]] = firrtl.object.subfield %A_object[domainInfo_in]
    // CHECK-NEXT: firrtl.propassign %[[domainInfo_in]], %A :
    // CHECK-NEXT: %[[associations_in:.+]] = firrtl.object.subfield %A_object[associations_in]
    // CHECK-NEXT: %[[path:.+]] = firrtl.path reference distinct[0]<>
    // CHECK-NEXT: %[[list:.+]] = firrtl.list.create %[[path]] :
    // CHECK-NEXT: firrtl.propassign %[[associations_in]], %[[list]]
    // CHECK-NEXT: firrtl.propassign %A_out, %A_object :
    //
    // CHECK-NOT: firrtl.wire
    //
    // CHECK:      %B_object = firrtl.object @ClockDomain_out
    // CHECK-NEXT: %[[domainInfo_in:.+]] = firrtl.object.subfield %B_object[domainInfo_in]
    // CHECK-NEXT: %[[associations_in:.+]] = firrtl.object.subfield %B_object[associations_in]
    // CHECK-NEXT: %[[path:.+]] = firrtl.path reference distinct[1]<>
    // CHECK-NEXT: %[[list:.+]] = firrtl.list.create %[[path]] :
    // CHECK-NEXT: firrtl.propassign %[[associations_in]], %[[list]]
    // CHECK-NEXT: firrtl.propassign %B_out, %B_object :
    //
    // CHECK-NEXT: firrtl.propassign %[[domainInfo_in]], %A :
    %w0 = firrtl.wire : !firrtl.domain
    %w1 = firrtl.wire : !firrtl.domain
    %w2 = firrtl.wire : !firrtl.domain
    firrtl.domain.define %w0, %A
    firrtl.domain.define %w1, %w0
    firrtl.domain.define %w2, %w1
    firrtl.domain.define %B, %w2
    // CHECK-NEXT: firrtl.matchingconnect %b, %a
    %0 = firrtl.unsafe_domain_cast %a domains %B : !firrtl.uint<1>
    firrtl.matchingconnect %b, %0 : !firrtl.uint<1>
    // CHECK-NOT: firrtl.wire
  }
}

// -----

// Test a "downards" U-turn.
firrtl.circuit "Foo" {
  firrtl.domain @ClockDomain
  firrtl.extmodule @Bar(
    in A: !firrtl.domain of @ClockDomain,
    in a: !firrtl.uint<1> domains [A]
  )
  firrtl.extmodule @Baz(
    out A: !firrtl.domain of @ClockDomain,
    out a: !firrtl.uint<1> domains [A]
  )
  // CHECK-LABEL: firrtl.module @Foo()
  firrtl.module @Foo() {
    // CHECK-NEXT: %bar_A, %bar_A_out, %bar_a = firrtl.instance bar @Bar
    %bar_A, %bar_a = firrtl.instance bar @Bar(
      in A: !firrtl.domain of @ClockDomain,
      in a: !firrtl.uint<1> domains [A]
    )
    // CHECK-NEXT: %baz_A_out, %baz_a = firrtl.instance baz @Baz
    %baz_A, %baz_a = firrtl.instance baz @Baz(
      out A: !firrtl.domain of @ClockDomain,
      out a: !firrtl.uint<1> domains [A]
    )
    // CHECK-NEXT: %[[#domainInfo_out:]] = firrtl.object.subfield %baz_A_out[domainInfo_out] :
    // CHECK-NEXT: firrtl.propassign %bar_A, %[[#domainInfo_out]] :
    firrtl.domain.define %bar_A, %baz_A
    // CHECK-NEXT: firrtl.matchingconnect %bar_a, %baz_a :
    firrtl.matchingconnect %bar_a, %baz_a : !firrtl.uint<1>
  }
}

// -----

// Test an anonymous domain.
firrtl.circuit "Foo" {
  firrtl.domain @ClockDomain
  firrtl.extmodule @Bar(
    in A: !firrtl.domain of @ClockDomain,
    in a: !firrtl.uint<1> domains [A]
  )
  // CHECK-LABEL: firrtl.module @Foo()
  firrtl.module @Foo() {
    // CHECK-NEXT: %bar_A, %bar_A_out, %bar_a = firrtl.instance bar @Bar
    %bar_A, %bar_a = firrtl.instance bar @Bar(
      in A: !firrtl.domain of @ClockDomain,
      in a: !firrtl.uint<1> domains [A]
    )
    %anon = firrtl.domain.anon : !firrtl.domain of @ClockDomain
    firrtl.domain.define %bar_A, %anon
    // CHECK-NOT: firrtl.object
    // CHECK-NOT: firrtl.domain.anon
    // CHECK-NOT: firrtl.domain.define
  }
}

// -----

// Check that unconnected output domains work.
//
// TODO: If there was stronger verification that ExpandWhens ensures
// static-single-connect, then this test could be illegal.  Currently, it is
// just here to make sure that this does something sensible.
firrtl.circuit "Foo" {
  firrtl.domain @ClockDomain
  // CHECK-LABEL: firrtl.module @Foo
  // CHECK-NOT:     in %A: !firrtl.class<@ClockDomain()>
  // CHECK-SAME:    out %A_out: !firrtl.class<@ClockDomain_out(
  firrtl.module @Foo(
    out %A: !firrtl.domain of @ClockDomain,
    in %a: !firrtl.uint<1> domains [%A]
  ) {
    // CHECK:      %A_object = firrtl.object @ClockDomain
    // CHECK-NEXT: %[[domainInfo_in:.+]] = firrtl.object.subfield %A_object[domainInfo_in]
    // CHECK-NEXT: %[[associations_in:.+]] = firrtl.object.subfield %A_object[associations_in]
    // CHECK-NEXT: %[[path:.+]] = firrtl.path reference distinct[0]<>
    // CHECK-NEXT: %[[list:.+]] = firrtl.list.create %[[path]] :
    // CHECK-NEXT: firrtl.propassign %[[associations_in]], %[[list]]
    // CHECK-NEXT: firrtl.propassign %A_out, %A_object :
  }
}

// -----

// Check the behavior of the lowering of an instance.
firrtl.circuit "Foo" {
  firrtl.domain @ClockDomain
  // CHECK-LABEL: firrtl.module @Bar(
  // CHECK-SAME:    in %A: !firrtl.class<@ClockDomain()>
  // CHECK-SAME:    out %A_out: !firrtl.class<@ClockDomain_out(
  // CHECK-SAME:    in %a: !firrtl.uint<1>
  firrtl.module @Bar(
    in %A: !firrtl.domain of @ClockDomain,
    in %a: !firrtl.uint<1> domains [%A]
  ) {}
  // CHECK-LABEL: firrtl.module @Foo(
  // CHECK-SAME:    in %A: !firrtl.class<@ClockDomain()>
  // CHECK-SAME:    out %A_out: !firrtl.class<@ClockDomain_out(
  // CHECK-SAME:    in %a: !firrtl.uint<1>
  firrtl.module @Foo(
    in %A: !firrtl.domain of @ClockDomain,
    in %a: !firrtl.uint<1> domains [%A]
  ) {
    // CHECK:      firrtl.instance bar @Bar(
    // CHECK-SAME:   in A: !firrtl.class<@ClockDomain()>
    // CHECK-SAME:   out A_out: !firrtl.class<@ClockDomain_out(
    // CHECK-SAME:   in a: !firrtl.uint<1>
    %bar_A, %bar_a = firrtl.instance bar @Bar(
      in A: !firrtl.domain of @ClockDomain,
      in a: !firrtl.uint<1> domains[A]
    )
    // CHECK:        firrtl.propassign %bar_A, %A
    firrtl.domain.define %bar_A, %A
    firrtl.matchingconnect %bar_a, %a : !firrtl.uint<1>
  }
}

// -----

// Check that multiple instances (port-to-instance) work.
firrtl.circuit "Foo" {
  firrtl.domain @ClockDomain
  firrtl.module @Bar(
    in %A: !firrtl.domain of @ClockDomain,
    in %a: !firrtl.uint<1> domains [%A]
  ) {}
  // CHECK-LABEL: firrtl.module @Foo(
  firrtl.module @Foo(
    in %A: !firrtl.domain of @ClockDomain,
    in %a: !firrtl.uint<1> domains [%A]
  ) {
    %bar1_A, %bar1_a = firrtl.instance bar1 @Bar(
      in A: !firrtl.domain of @ClockDomain,
      in a: !firrtl.uint<1> domains[A]
    )
    // CHECK-NOT:    firrtl.domain.define
    // CHECK:        firrtl.propassign %bar1_A, %A
    firrtl.domain.define %bar1_A, %A
    firrtl.matchingconnect %bar1_a, %a : !firrtl.uint<1>
    %bar2_A, %bar2_a = firrtl.instance bar2 @Bar(
      in A: !firrtl.domain of @ClockDomain,
      in a: !firrtl.uint<1> domains[A]
    )
    // CHECK-NOT:    firrtl.domain.define
    // CHECK:        firrtl.propassign %bar2_A, %A
    // CHECK-NOT:    firrtl.domain.define
    firrtl.domain.define %bar2_A, %A
    firrtl.matchingconnect %bar2_a, %a : !firrtl.uint<1>
  }
}

// -----

// Check that fan-out from one instance to many domain ports works.
firrtl.circuit "Foo" {
  firrtl.domain @ClockDomain
  firrtl.extmodule @Bar(
    out A: !firrtl.domain of @ClockDomain,
    out a: !firrtl.uint<1> domains [A]
  )
  // CHECK-LABEL: firrtl.module @Foo(
  firrtl.module @Foo(
    out %A: !firrtl.domain of @ClockDomain,
    out %a: !firrtl.uint<1> domains [%A],
    out %B: !firrtl.domain of @ClockDomain,
    out %b: !firrtl.uint<1> domains [%B]
  ) {
    %bar_A, %bar_a = firrtl.instance bar @Bar(
      out A: !firrtl.domain of @ClockDomain,
      out a: !firrtl.uint<1> domains[A]
    )
    firrtl.domain.define %A, %bar_A
    firrtl.domain.define %B, %bar_A
    firrtl.matchingconnect %a, %bar_a : !firrtl.uint<1>
    firrtl.matchingconnect %b, %bar_a : !firrtl.uint<1>
    // CHECK:      %[[A_domainInfo_in:.+]] = firrtl.object.subfield %A_object[domainInfo_in]
    // CHECK:      %[[B_domainInfo_in:.+]] = firrtl.object.subfield %B_object[domainInfo_in]
    // CHECK:      %[[bar_domainInfo_out:.+]] = firrtl.object.subfield %bar_A_out[domainInfo_out] :
    // CHECK-NEXT: firrtl.propassign %[[A_domainInfo_in]], %[[bar_domainInfo_out]]
    // CHECK-NEXT: firrtl.propassign %[[B_domainInfo_in]], %[[bar_domainInfo_out]]
  }
}

// -----

// Check the behavior of external modules.
firrtl.circuit "Foo" {
  firrtl.domain @ClockDomain
  // CHECK-LABEL: firrtl.extmodule @Bar(
  // CHECK-SAME:    in A: !firrtl.class<@ClockDomain()>
  // CHECK-SAME:    out A_out: !firrtl.class<@ClockDomain_out(
  // CHECK-SAME:    in a: !firrtl.uint<1>
  firrtl.extmodule @Bar(
    in A: !firrtl.domain of @ClockDomain,
    in a: !firrtl.uint<1> domains [A]
  )
  firrtl.module @Foo(
  ) {}
}

// -----

// Check that input/output properties and domain bodies are copied over.
firrtl.circuit "Foo" {
  // CHECK-LABEL: firrtl.class @ClockDomain(
  // CHECK-SAME:    in %name_in: !firrtl.string,
  // CHECK-SAME:    out %name_out: !firrtl.string
  // CHECK-SAME:  )
  // CHECK-NEXT:    firrtl.propassign %name_out, %name_in : !firrtl.string
  firrtl.domain @ClockDomain [
    #firrtl.domain.field<"name", !firrtl.string>
  ]
  firrtl.module @Foo() {}
}

// -----

// Check that unsafe domain casts are dropped.
firrtl.circuit "Foo" {
  firrtl.domain @ClockDomain
  // CHECK-LABEL: firrtl.module @Foo
  firrtl.module @Foo(
    in %A: !firrtl.domain of @ClockDomain,
    in %a: !firrtl.uint<1> domains [%A],
    in %B: !firrtl.domain of @ClockDomain,
    out %b: !firrtl.uint<1> domains [%B]
  ) {
    // CHECK-NOT: firrtl.unsafe_domain_cast
    // CHECK: firrtl.matchingconnect %b, %a
    // CHECK-NOT: firrtl.unsafe_domain_cast
    %0 = firrtl.unsafe_domain_cast %a domains %B : !firrtl.uint<1>
    firrtl.matchingconnect %b, %0 : !firrtl.uint<1>
  }
}

// -----

// Check that dead/unused operations inolving domains are deleted.
firrtl.circuit "DeadDomainOps" {
  firrtl.domain @ClockDomain
  // CHECK-LABEL: firrtl.module @DeadDomainOps
  // CHECK-NOT: firrtl.wire
  // CHECK-NOT: firrtl.domain.define
  // CHECK-NOT: firrtl.domain.anon : !firrtl.domain of @ClockDomain
  firrtl.module @DeadDomainOps(
  ) {
    // A lone, undriven wire.
    %a = firrtl.wire : !firrtl.domain

    // One wire that drives another.
    %b = firrtl.wire : !firrtl.domain
    %c = firrtl.wire : !firrtl.domain
    firrtl.domain.define %b, %c

    // A lone anonymous domain.
    %d = firrtl.domain.anon : !firrtl.domain of @ClockDomain

    // A wire driven by an anonymous domain.
    %e = firrtl.wire : !firrtl.domain
    %f = firrtl.domain.anon : !firrtl.domain of @ClockDomain
    firrtl.domain.define %e, %f
  }
}

// -----

// Test that zero width ports that have domain associations do not have these
// associations propagated to the lowered class for that domain port.  If these
// did, then this would create symbols on zero-width ports which is not allowed
// by the LowerToHW conversion.
firrtl.circuit "ZeroWidthPort" {
  firrtl.domain @ClockDomain
  // CHECK-LABEL: firrtl.module @ZeroWidthPort(
  firrtl.module @ZeroWidthPort(
    in %A: !firrtl.domain of @ClockDomain,
    in %a: !firrtl.uint<0> domains [%A]
  ) {
    // CHECK:       %[[associations_in:.+]] = firrtl.object.subfield %A_object[associations_in]
    // CHECK-NEXT: %[[list:.+]] = firrtl.list.create :
    // CHECK-NEXT: firrtl.propassign %[[associations_in]], %[[list]]
    // CHECK-NEXT: firrtl.propassign %A_out, %A_object :
  }
}
