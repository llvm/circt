// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-domains))' --split-input-file %s | FileCheck %s

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
  firrtl.domain @ClockDomain {}
  // CHECK-LABEL: firrtl.module @Foo(
  // CHECK-NOT:     firrtl.domain
  // CHECK-SAME:    in %A: !firrtl.class<@ClockDomain()>
  // CHECK-NOT:     firrtl.domain
  // CHECK-SAME:    out %A_out: !firrtl.class<@ClockDomain_out(
  // CHECK-SAME:    in %a: !firrtl.uint<1> [{class = "circt.tracker", id = distinct[0]<>}]
  // CHECK-NOT:       domains
  firrtl.module @Foo(
    in %A: !firrtl.domain of @ClockDomain,
    in %a: !firrtl.uint<1> domains [%A]
    // CHECK:      %A_object = firrtl.object @ClockDomain
    // CHECK-NEXT: %[[domainInfo_in:.+]] = firrtl.object.subfield %A_object[domainInfo_in]
    // CHECK-NEXT: firrtl.propassign %[[domainInfo_in]], %A :
    // CHECK-NEXT: %[[associations_in:.+]] = firrtl.object.subfield %A_object[associations_in]
    // CHECK-NEXT: %[[path:.+]] = firrtl.path member_reference distinct[0]<>
    // CHECK-NEXT: %[[list:.+]] = firrtl.list.create %[[path]] :
    // CHECK-NEXT: firrtl.propassign %[[associations_in]], %[[list]]
    // CHECK-NEXT: firrtl.propassign %A_out, %A_object :
  ) {}
}

// -----

firrtl.circuit "Foo" {
  firrtl.domain @ClockDomain {}
  // The external module should have all domain ports erased and domain
  // associations removed.
  //
  // CHECK-LABEL: firrtl.extmodule @Bar(
  // CHECK-NOT:     firrtl.domain
  // CHECK-SAME:    in a: !firrtl.uint<1>
  // CHECK-NOT:       domains
  firrtl.extmodule @Bar(
    in A: !firrtl.domain of @ClockDomain,
    in a: !firrtl.uint<1> domains [A]
  )
  // CHECK-LABEL: firrtl.module @Foo(
  firrtl.module @Foo(
    in %A: !firrtl.domain of @ClockDomain,
    in %a: !firrtl.uint<1> domains [%A]
  ) {
    %bar_A, %bar_a = firrtl.instance bar @Bar(
      in A: !firrtl.domain of @ClockDomain,
      in a: !firrtl.uint<1> domains[A]
    )
    firrtl.matchingconnect %bar_a, %a : !firrtl.uint<1>
  }
}
