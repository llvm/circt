// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-domains))' --split-input-file %s | FileCheck --implicit-check-not firrtl.domain %s

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
  firrtl.domain @ClockDomain() {}
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
  firrtl.domain @ClockDomain() {}
  // CHECK-LABEL: firrtl.class @ResetDomain(
  // CHECK-LABEL: firrtl.class @ResetDomain_out(
  // CHECK-SAME:    in %domainInfo_in: !firrtl.class<@ResetDomain()>
  // CHECK-SAME:    out %domainInfo_out: !firrtl.class<@ResetDomain()>
  // CHECK-SAME:    in %associations_in: !firrtl.list<path>
  // CHECK-SAME:    out %associations_out: !firrtl.list<path>
  firrtl.domain @ResetDomain() {}
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
  firrtl.domain @ClockDomain() {}
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

// Check that output domains work.
//
// TODO: This is currently insufficient as we don't yet have domain connection
// operations.
firrtl.circuit "Foo" {
  firrtl.domain @ClockDomain() {}
  // CHECK-LABEL: firrtl.module @Foo
  // CHECK-NOT:     in %A: !firrtl.class<@ClockDomain()>
  // CHECK-SAME:    out %A_out: !firrtl.class<@ClockDomain_out(
  firrtl.module @Foo(
    out %A: !firrtl.domain of @ClockDomain,
    in %a: !firrtl.uint<1> domains [%A]
  ) {
    // CHECK:      %A_object = firrtl.object @ClockDomain
    // CHECK-NOT:  firrtl.object.subfield %A_object[domainInfo_in]
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
  firrtl.domain @ClockDomain() {}
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
    firrtl.matchingconnect %bar_a, %a : !firrtl.uint<1>
  }
}

// -----

// Check the behavior of external modules.
firrtl.circuit "Foo" {
  firrtl.domain @ClockDomain() {}
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
  firrtl.domain @ClockDomain(
    in %name_in: !firrtl.string,
    out %name_out: !firrtl.string
  ) {
    firrtl.propassign %name_out, %name_in : !firrtl.string
  }
  firrtl.module @Foo() {}
}
