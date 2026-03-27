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
    in %A: !firrtl.domain<@ClockDomain()>,
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
    in %A: !firrtl.domain<@ClockDomain()>,
    in %B: !firrtl.domain<@ResetDomain()>,
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
    in %A: !firrtl.domain<@ClockDomain()>,
    in %B: !firrtl.domain<@ClockDomain()>,
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
    in %A: !firrtl.domain<@ClockDomain()>,
    in %a0: !firrtl.uint<1> domains [%A],
    in %a1: !firrtl.uint<1> domains [%A],
    in %a2: !firrtl.uint<1> domains [%A],
    in %B: !firrtl.domain<@ClockDomain()>,
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
    in %A: !firrtl.domain<@ClockDomain()>,
    out %B: !firrtl.domain<@ClockDomain()>,
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
    firrtl.domain.define %B, %A : !firrtl.domain<@ClockDomain()>
    // CHECK-NEXT: firrtl.matchingconnect %b, %a
    %0 = firrtl.unsafe_domain_cast %a domains[%B] : !firrtl.uint<1> domains[!firrtl.domain<@ClockDomain()>]
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
    in %A: !firrtl.domain<@ClockDomain()>,
    out %B: !firrtl.domain<@ClockDomain()>,
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
    %w0 = firrtl.wire : !firrtl.domain<@ClockDomain()>
    %w1 = firrtl.wire : !firrtl.domain<@ClockDomain()>
    %w2 = firrtl.wire : !firrtl.domain<@ClockDomain()>
    firrtl.domain.define %w0, %A : !firrtl.domain<@ClockDomain()>
    firrtl.domain.define %w1, %w0 : !firrtl.domain<@ClockDomain()>
    firrtl.domain.define %w2, %w1 : !firrtl.domain<@ClockDomain()>
    firrtl.domain.define %B, %w2 : !firrtl.domain<@ClockDomain()>
    // CHECK-NEXT: firrtl.matchingconnect %b, %a
    %0 = firrtl.unsafe_domain_cast %a domains[%B] : !firrtl.uint<1> domains[!firrtl.domain<@ClockDomain()>]
    firrtl.matchingconnect %b, %0 : !firrtl.uint<1>
    // CHECK-NOT: firrtl.wire
  }
}

// -----

// Test that wires with multiple destination users works.  This avoids a
// potential coding bug that assumes wires only have one destination.
firrtl.circuit "Foo" {
  firrtl.domain @ClockDomain
  firrtl.extmodule @Bar(
    in a: !firrtl.domain<@ClockDomain()>,
    in b: !firrtl.domain<@ClockDomain()>
  )
  // CHECK-LABEL: firrtl.module @Foo
  firrtl.module @Foo(
    in %a: !firrtl.domain<@ClockDomain()>
  ) {
    %bar_a, %bar_b = firrtl.instance bar @Bar(
      in a: !firrtl.domain<@ClockDomain()>,
      in b: !firrtl.domain<@ClockDomain()>
    )
    %wire = firrtl.wire : !firrtl.domain<@ClockDomain()>
    firrtl.domain.define %wire, %a : !firrtl.domain<@ClockDomain()>
    // CHECK:      firrtl.propassign %bar_a, %a
    // CHECK-NEXT: firrtl.propassign %bar_b, %a
    firrtl.domain.define %bar_a, %wire : !firrtl.domain<@ClockDomain()>
    firrtl.domain.define %bar_b, %wire : !firrtl.domain<@ClockDomain()>
  }
}

// -----

// Test a "downards" U-turn.
firrtl.circuit "Foo" {
  firrtl.domain @ClockDomain
  firrtl.extmodule @Bar(
    in A: !firrtl.domain<@ClockDomain()>,
    in a: !firrtl.uint<1> domains [A]
  )
  firrtl.extmodule @Baz(
    out A: !firrtl.domain<@ClockDomain()>,
    out a: !firrtl.uint<1> domains [A]
  )
  // CHECK-LABEL: firrtl.module @Foo()
  firrtl.module @Foo() {
    // CHECK-NEXT: %bar_A, %bar_A_out, %bar_a = firrtl.instance bar @Bar
    %bar_A, %bar_a = firrtl.instance bar @Bar(
      in A: !firrtl.domain<@ClockDomain()>,
      in a: !firrtl.uint<1> domains [A]
    )
    // CHECK-NEXT: %baz_A_out, %baz_a = firrtl.instance baz @Baz
    %baz_A, %baz_a = firrtl.instance baz @Baz(
      out A: !firrtl.domain<@ClockDomain()>,
      out a: !firrtl.uint<1> domains [A]
    )
    // CHECK-NEXT: %[[#domainInfo_out:]] = firrtl.object.subfield %baz_A_out[domainInfo_out] :
    // CHECK-NEXT: firrtl.propassign %bar_A, %[[#domainInfo_out]] :
    firrtl.domain.define %bar_A, %baz_A : !firrtl.domain<@ClockDomain()>
    // CHECK-NEXT: firrtl.matchingconnect %bar_a, %baz_a :
    firrtl.matchingconnect %bar_a, %baz_a : !firrtl.uint<1>
  }
}

// -----

// Test an anonymous domain.
firrtl.circuit "Foo" {
  firrtl.domain @ClockDomain
  firrtl.extmodule @Bar(
    in A: !firrtl.domain<@ClockDomain()>,
    in a: !firrtl.uint<1> domains [A]
  )
  // CHECK-LABEL: firrtl.module @Foo()
  firrtl.module @Foo() {
    // CHECK-NEXT: %bar_A, %bar_A_out, %bar_a = firrtl.instance bar @Bar
    %bar_A, %bar_a = firrtl.instance bar @Bar(
      in A: !firrtl.domain<@ClockDomain()>,
      in a: !firrtl.uint<1> domains [A]
    )
    // CHECK-NEXT: %[[#UNKNOWN:]] = firrtl.unknown : !firrtl.class<@ClockDomain()>
    %anon = firrtl.domain.anon : !firrtl.domain<@ClockDomain()>
    // CHECK-NEXT: firrtl.propassign %bar_A, %[[#UNKNOWN]]
    firrtl.domain.define %bar_A, %anon : !firrtl.domain<@ClockDomain()>
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
    out %A: !firrtl.domain<@ClockDomain()>,
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
    in %A: !firrtl.domain<@ClockDomain()>,
    in %a: !firrtl.uint<1> domains [%A]
  ) {}
  // CHECK-LABEL: firrtl.module @Foo(
  // CHECK-SAME:    in %A: !firrtl.class<@ClockDomain()>
  // CHECK-SAME:    out %A_out: !firrtl.class<@ClockDomain_out(
  // CHECK-SAME:    in %a: !firrtl.uint<1>
  firrtl.module @Foo(
    in %A: !firrtl.domain<@ClockDomain()>,
    in %a: !firrtl.uint<1> domains [%A]
  ) {
    // CHECK:      firrtl.instance bar @Bar(
    // CHECK-SAME:   in A: !firrtl.class<@ClockDomain()>
    // CHECK-SAME:   out A_out: !firrtl.class<@ClockDomain_out(
    // CHECK-SAME:   in a: !firrtl.uint<1>
    %bar_A, %bar_a = firrtl.instance bar @Bar(
      in A: !firrtl.domain<@ClockDomain()>,
      in a: !firrtl.uint<1> domains[A]
    )
    // CHECK:        firrtl.propassign %bar_A, %A
    firrtl.domain.define %bar_A, %A : !firrtl.domain<@ClockDomain()>
    firrtl.matchingconnect %bar_a, %a : !firrtl.uint<1>
  }
}

// -----

// Check that multiple instances (port-to-instance) work.
firrtl.circuit "Foo" {
  firrtl.domain @ClockDomain
  firrtl.module @Bar(
    in %A: !firrtl.domain<@ClockDomain()>,
    in %a: !firrtl.uint<1> domains [%A]
  ) {}
  // CHECK-LABEL: firrtl.module @Foo(
  firrtl.module @Foo(
    in %A: !firrtl.domain<@ClockDomain()>,
    in %a: !firrtl.uint<1> domains [%A]
  ) {
    %bar1_A, %bar1_a = firrtl.instance bar1 @Bar(
      in A: !firrtl.domain<@ClockDomain()>,
      in a: !firrtl.uint<1> domains[A]
    )
    // CHECK-NOT:    firrtl.domain.define
    // CHECK:        firrtl.propassign %bar1_A, %A
    firrtl.domain.define %bar1_A, %A : !firrtl.domain<@ClockDomain()>
    firrtl.matchingconnect %bar1_a, %a : !firrtl.uint<1>
    %bar2_A, %bar2_a = firrtl.instance bar2 @Bar(
      in A: !firrtl.domain<@ClockDomain()>,
      in a: !firrtl.uint<1> domains[A]
    )
    // CHECK-NOT:    firrtl.domain.define
    // CHECK:        firrtl.propassign %bar2_A, %A
    // CHECK-NOT:    firrtl.domain.define
    firrtl.domain.define %bar2_A, %A : !firrtl.domain<@ClockDomain()>
    firrtl.matchingconnect %bar2_a, %a : !firrtl.uint<1>
  }
}

// -----

// Check that fan-out from one instance to many domain ports works.
firrtl.circuit "Foo" {
  firrtl.domain @ClockDomain
  firrtl.extmodule @Bar(
    out A: !firrtl.domain<@ClockDomain()>,
    out a: !firrtl.uint<1> domains [A]
  )
  // CHECK-LABEL: firrtl.module @Foo(
  firrtl.module @Foo(
    out %A: !firrtl.domain<@ClockDomain()>,
    out %a: !firrtl.uint<1> domains [%A],
    out %B: !firrtl.domain<@ClockDomain()>,
    out %b: !firrtl.uint<1> domains [%B]
  ) {
    %bar_A, %bar_a = firrtl.instance bar @Bar(
      out A: !firrtl.domain<@ClockDomain()>,
      out a: !firrtl.uint<1> domains[A]
    )
    firrtl.domain.define %A, %bar_A : !firrtl.domain<@ClockDomain()>
    firrtl.domain.define %B, %bar_A : !firrtl.domain<@ClockDomain()>
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
    in A: !firrtl.domain<@ClockDomain()>,
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
    in %A: !firrtl.domain<@ClockDomain()>,
    in %a: !firrtl.uint<1> domains [%A],
    in %B: !firrtl.domain<@ClockDomain()>,
    out %b: !firrtl.uint<1> domains [%B]
  ) {
    // CHECK-NOT: firrtl.unsafe_domain_cast
    // CHECK: firrtl.matchingconnect %b, %a
    // CHECK-NOT: firrtl.unsafe_domain_cast
    %0 = firrtl.unsafe_domain_cast %a domains[%B] : !firrtl.uint<1> domains[!firrtl.domain<@ClockDomain()>]
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
  // CHECK-NOT: firrtl.domain.anon : !firrtl.domain<@ClockDomain()>
  // CHECK-NOT: firrtl.unknown
  firrtl.module @DeadDomainOps(
  ) {
    // A lone, undriven wire.
    %a = firrtl.wire : !firrtl.domain<@ClockDomain()>

    // One wire that drives another.
    %b = firrtl.wire : !firrtl.domain<@ClockDomain()>
    %c = firrtl.wire : !firrtl.domain<@ClockDomain()>
    firrtl.domain.define %b, %c : !firrtl.domain<@ClockDomain()>

    // A lone anonymous domain.
    %d = firrtl.domain.anon : !firrtl.domain<@ClockDomain()>

    // A wire driven by an anonymous domain.
    %e = firrtl.wire : !firrtl.domain<@ClockDomain()>
    %f = firrtl.domain.anon : !firrtl.domain<@ClockDomain()>
    firrtl.domain.define %e, %f : !firrtl.domain<@ClockDomain()>
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
    in %A: !firrtl.domain<@ClockDomain()>,
    in %a: !firrtl.uint<0> domains [%A]
  ) {
    // CHECK:      %[[associations_in:.+]] = firrtl.object.subfield %A_object[associations_in]
    // CHECK-NEXT: %[[list:.+]] = firrtl.list.create :
    // CHECK-NEXT: firrtl.propassign %[[associations_in]], %[[list]]
    // CHECK-NEXT: firrtl.propassign %A_out, %A_object :
  }
}

// -----

// Test that domain create operations are properly lowered to object
// instantiations.
firrtl.circuit "DomainCreate" {
  // CHECK-LABEL: firrtl.class @ClockDomain(
  // CHECK-LABEL: firrtl.class @ClockDomain_out(
  firrtl.domain @ClockDomain
  // CHECK-LABEL: firrtl.module @DomainCreate(
  // CHECK-SAME:    out %A_out: !firrtl.class<@ClockDomain_out(
  firrtl.module @DomainCreate(
    out %A: !firrtl.domain<@ClockDomain()>
  ) {
    // CHECK:      %A_object = firrtl.object @ClockDomain_out
    // CHECK-NEXT: %[[domainInfo_in:.+]] = firrtl.object.subfield %A_object[domainInfo_in]
    // CHECK:      %[[associations_in:.+]] = firrtl.object.subfield %A_object[associations_in]
    // CHECK-NEXT: %[[list:.+]] = firrtl.list.create :
    // CHECK-NEXT: firrtl.propassign %[[associations_in]], %[[list]]
    // CHECK-NEXT: firrtl.propassign %A_out, %A_object :
    // CHECK-NEXT: %my_domain = firrtl.object @ClockDomain()
    // CHECK-NEXT: firrtl.propassign %[[domainInfo_in]], %my_domain
    %my_domain = firrtl.domain.create : !firrtl.domain<@ClockDomain()>
    firrtl.domain.define %A, %my_domain : !firrtl.domain<@ClockDomain()>
  }
}

// -----

// Test that dead domain create operations are erased.
firrtl.circuit "DeadDomainCreate" {
  firrtl.domain @ClockDomain
  // CHECK-LABEL: firrtl.module @DeadDomainCreate() {
  // CHECK-NEXT:  }
  firrtl.module @DeadDomainCreate() {
    %my_domain = firrtl.domain.create : !firrtl.domain<@ClockDomain()>
  }
}

// -----

// Test domain create with mixed constant and port values.
firrtl.circuit "DomainCreateWithMixedValues" {
  firrtl.domain @ClockDomain [#firrtl.domain.field<"name", !firrtl.string>, #firrtl.domain.field<"period", !firrtl.integer>]
  // CHECK-LABEL: firrtl.module @DomainCreateWithMixedValues(
  firrtl.module @DomainCreateWithMixedValues(
    in %clk_period: !firrtl.integer,
    out %A: !firrtl.domain<@ClockDomain(name: !firrtl.string, period: !firrtl.integer)>
  ) {
    // CHECK:      %[[name_const:.+]] = firrtl.string "FastClock"
    // CHECK-NEXT: %my_domain = firrtl.object @ClockDomain(
    // CHECK-NEXT: %[[name_in:.+]] = firrtl.object.subfield %my_domain[name_in]
    // CHECK-NEXT: firrtl.propassign %[[name_in]], %[[name_const]]
    // CHECK-NEXT: %[[period_in:.+]] = firrtl.object.subfield %my_domain[period_in]
    // CHECK-NEXT: firrtl.propassign %[[period_in]], %clk_period
    %name = firrtl.string "FastClock"
    %my_domain = firrtl.domain.create(%name, %clk_period) : !firrtl.domain<@ClockDomain(name: !firrtl.string, period: !firrtl.integer)>
    firrtl.domain.define %A, %my_domain : !firrtl.domain<@ClockDomain(name: !firrtl.string, period: !firrtl.integer)>
  }
}

// -----

// Test that domain subfield operations are properly lowered to object subfield.
// This tests both used and unused domain subfield operations.
firrtl.circuit "DomainSubfield" {
  firrtl.domain @ClockDomain [
    #firrtl.domain.field<"a", !firrtl.integer>,
    #firrtl.domain.field<"b", !firrtl.integer>,
    #firrtl.domain.field<"c", !firrtl.integer>,
    #firrtl.domain.field<"d", !firrtl.integer>
  ]
  // CHECK-LABEL: firrtl.module @DomainSubfield(
  firrtl.module @DomainSubfield(
    in %A: !firrtl.domain<@ClockDomain(a: !firrtl.integer, b: !firrtl.integer, c: !firrtl.integer, d: !firrtl.integer)>,
    out %B: !firrtl.domain<@ClockDomain(a: !firrtl.integer, b: !firrtl.integer, c: !firrtl.integer, d: !firrtl.integer)>
  ) {
    // CHECK:      %[[a_out:.+]] = firrtl.object.subfield %A[a_out]
    // CHECK:      %[[b_out:.+]] = firrtl.object.subfield %A[b_out]
    // CHECK:      %[[c_out:.+]] = firrtl.object.subfield %A[c_out]
    // CHECK:      %[[d_out:.+]] = firrtl.object.subfield %A[d_out]
    // CHECK:      %C = firrtl.object @ClockDomain
    // CHECK:      %[[a_in:.+]] = firrtl.object.subfield %C[a_in]
    // CHECK-NEXT: firrtl.propassign %[[a_in]], %[[a_out]]
    // CHECK:      %[[b_in:.+]] = firrtl.object.subfield %C[b_in]
    // CHECK-NEXT: firrtl.propassign %[[b_in]], %[[b_out]]
    // CHECK:      %[[c_in:.+]] = firrtl.object.subfield %C[c_in]
    // CHECK-NEXT: firrtl.propassign %[[c_in]], %[[c_out]]
    // CHECK:      %[[d_in:.+]] = firrtl.object.subfield %C[d_in]
    // CHECK-NEXT: firrtl.propassign %[[d_in]], %[[d_out]]
    // CHECK:      %[[unused_a:.+]] = firrtl.object.subfield %A[a_out]
    // CHECK:      %[[unused_c:.+]] = firrtl.object.subfield %A[c_out]
    %a = firrtl.domain.subfield %A["a"] : !firrtl.domain<@ClockDomain(a: !firrtl.integer, b: !firrtl.integer, c: !firrtl.integer, d: !firrtl.integer)>
    %b = firrtl.domain.subfield %A["b"] : !firrtl.domain<@ClockDomain(a: !firrtl.integer, b: !firrtl.integer, c: !firrtl.integer, d: !firrtl.integer)>
    %c = firrtl.domain.subfield %A["c"] : !firrtl.domain<@ClockDomain(a: !firrtl.integer, b: !firrtl.integer, c: !firrtl.integer, d: !firrtl.integer)>
    %d = firrtl.domain.subfield %A["d"] : !firrtl.domain<@ClockDomain(a: !firrtl.integer, b: !firrtl.integer, c: !firrtl.integer, d: !firrtl.integer)>
    %C = firrtl.domain.create(%a, %b, %c, %d) : !firrtl.domain<@ClockDomain(a: !firrtl.integer, b: !firrtl.integer, c: !firrtl.integer, d: !firrtl.integer)>
    firrtl.domain.define %B, %C : !firrtl.domain<@ClockDomain(a: !firrtl.integer, b: !firrtl.integer, c: !firrtl.integer, d: !firrtl.integer)>
    %unused_a = firrtl.domain.subfield %A["a"] : !firrtl.domain<@ClockDomain(a: !firrtl.integer, b: !firrtl.integer, c: !firrtl.integer, d: !firrtl.integer)>
    %unused_c = firrtl.domain.subfield %A["c"] : !firrtl.domain<@ClockDomain(a: !firrtl.integer, b: !firrtl.integer, c: !firrtl.integer, d: !firrtl.integer)>
  }
}

// -----

// Test that domain operands on a non-domain-type wire are erased during
// lowering, leaving a clean wire.  The domain port lowering is orthogonal;
// we just verify the wire line is free of domain operands.
firrtl.circuit "WireWithPortDomain" {
  firrtl.domain @ClockDomain
  // CHECK-LABEL: firrtl.module @WireWithPortDomain
  // CHECK-NOT:   firrtl.module
  // CHECK:         %w = firrtl.wire : !firrtl.uint<1>
  // CHECK-NEXT:    firrtl.matchingconnect %w, %a
  firrtl.module @WireWithPortDomain(
    in %D: !firrtl.domain<@ClockDomain()>,
    in %a: !firrtl.uint<1> domains [%D]
  ) {
    %w = firrtl.wire domains[%D] : !firrtl.uint<1> domains[!firrtl.domain<@ClockDomain()>]
    firrtl.matchingconnect %w, %a : !firrtl.uint<1>
  }
}

// -----

// Test that domain operands from a domain.create op leave no stray conversion
// casts after lowering.
firrtl.circuit "WireWithCreateDomain" {
  firrtl.domain @ClockDomain
  // CHECK-LABEL: firrtl.module @WireWithCreateDomain()
  // CHECK-NEXT:    %my_domain = firrtl.object @ClockDomain()
  // CHECK-NEXT:    %w = firrtl.wire : !firrtl.uint<1>
  firrtl.module @WireWithCreateDomain() {
    %my_domain = firrtl.domain.create : !firrtl.domain<@ClockDomain()>
    %w = firrtl.wire domains[%my_domain] : !firrtl.uint<1> domains[!firrtl.domain<@ClockDomain()>]
  }
}
