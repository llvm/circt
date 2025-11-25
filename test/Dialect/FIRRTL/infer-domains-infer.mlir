// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-infer-domains{mode=infer}))' %s | FileCheck %s

firrtl.circuit "InferOutputDomain" {
  firrtl.domain @ClockDomain

  firrtl.extmodule @Foo(out D: !firrtl.domain of @ClockDomain, out x: !firrtl.uint<1> domains [D])

  // CHECK: firrtl.module private @Bar(out %ClockDomain: !firrtl.domain of @ClockDomain, out %x: !firrtl.uint<1> domains [%ClockDomain]) {
  // CHECK:   %foo_D, %foo_x = firrtl.instance foo @Foo(out D: !firrtl.domain of @ClockDomain, out x: !firrtl.uint<1> domains [D])
  // CHECK:   firrtl.matchingconnect %x, %foo_x : !firrtl.uint<1>
  // CHECK:   firrtl.domain.define %ClockDomain, %foo_D
  // CHECK: }
  firrtl.module private @Bar(out %x : !firrtl.uint<1>) {
    %foo_D, %foo_x = firrtl.instance foo @Foo(out D: !firrtl.domain of @ClockDomain, out x: !firrtl.uint<1> domains [D])
    firrtl.matchingconnect %x, %foo_x : !firrtl.uint<1>
  }

  // CHECK: firrtl.module @InferOutputDomain(out %D: !firrtl.domain of @ClockDomain, out %x: !firrtl.uint<1> domains [%D]) {
  // CHECK:   %bar_ClockDomain, %bar_x = firrtl.instance bar @Bar(out ClockDomain: !firrtl.domain of @ClockDomain, out x: !firrtl.uint<1> domains [ClockDomain])
  // CHECK:   firrtl.matchingconnect %x, %bar_x : !firrtl.uint<1>
  // CHECK:   firrtl.domain.define %D, %bar_ClockDomain
  // CHECK: }
  firrtl.module @InferOutputDomain(out %D: !firrtl.domain of @ClockDomain, out %x: !firrtl.uint<1> domains [%D]) {
    %bar_x = firrtl.instance bar @Bar(out x : !firrtl.uint<1>)
    firrtl.matchingconnect %x, %bar_x : !firrtl.uint<1>
  }
}

// do not crash the InferDomains pass.  This stems from the fact that "no domain
// information" can be represented as both an empty array `[]` and an empty
// array of arrays `[[]]`.
firrtl.circuit "EmptyDomainInfo" {
  firrtl.domain @DomainKind
  firrtl.module @EmptyDomainInfo(out %x: !firrtl.integer) {
    %0 = firrtl.integer 5
    firrtl.propassign %x, %0 : !firrtl.integer
  }
}