// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-infer-domains{mode=infer}))' %s | FileCheck %s

firrtl.circuit "InferOutputDomain" {
  firrtl.domain @ClockDomain

  firrtl.extmodule @Foo(out D: !firrtl.domain of @ClockDomain, out x: !firrtl.uint<1> domains [D])

  // CHECK: firrtl.module private @Bar(out %ClockDomain: !firrtl.domain of @ClockDomain, out %x: !firrtl.uint<1> domains [%ClockDomain]) {
  // CHECK:   %foo_D, %foo_x = firrtl.instance foo @Foo(out D: !firrtl.domain of @ClockDomain, out x: !firrtl.uint<1> domains [D])
  // CHECK:   firrtl.domain.define %ClockDomain, %foo_D
  // CHECK:   firrtl.matchingconnect %x, %foo_x : !firrtl.uint<1>
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

// CHECK-LABEL: UndrivenInstanceDomainPort
firrtl.circuit "UndrivenInstanceDomainPort" {
  firrtl.domain @ClockDomain

  firrtl.extmodule @Foo(in c : !firrtl.domain of @ClockDomain)

  // CHECK: firrtl.module @UndrivenInstanceDomainPort() {
  // CHECK:   %foo_c = firrtl.instance foo @Foo(in c: !firrtl.domain of @ClockDomain)
  // CHECK:   %0 = firrtl.domain.anon : !firrtl.domain of @ClockDomain
  // CHECK:   firrtl.domain.define %foo_c, %0
  // CHECK: }
  firrtl.module @UndrivenInstanceDomainPort() {
    %foo_c = firrtl.instance foo @Foo(in c : !firrtl.domain of @ClockDomain)
  }
}

// CHECK-LABEL: UndrivenInstanceChoiceDomainPort
firrtl.circuit "UndrivenInstanceChoiceDomainPort" {
  firrtl.domain @ClockDomain

  firrtl.option @Option {
    firrtl.option_case @X
  }

  firrtl.extmodule @Foo(in c : !firrtl.domain of @ClockDomain)
  firrtl.extmodule @Bar(in c : !firrtl.domain of @ClockDomain)

  // CHECK: firrtl.module @UndrivenInstanceChoiceDomainPort() {
  // CHECK:   %inst_c = firrtl.instance_choice inst @Foo alternatives @Option { @X -> @Bar } (in c: !firrtl.domain of @ClockDomain)
  // CHECK:   %0 = firrtl.domain.anon : !firrtl.domain of @ClockDomain
  // CHECK:   firrtl.domain.define %inst_c, %0
  // CHECK: }
  firrtl.module @UndrivenInstanceChoiceDomainPort() {
    %inst_c = firrtl.instance_choice inst @Foo alternatives @Option { @X -> @Bar } (in c : !firrtl.domain of @ClockDomain)
  }
}

// CHECK-LABEL: UnableToInferDomainOfPortDrivenByConstant
firrtl.circuit "UnableToInferDomainOfPortDrivenByConstant" {
  firrtl.domain @ClockDomain
  firrtl.module private @Foo(in %i: !firrtl.uint<1>) {}
  firrtl.module private @Bar(in %i: !firrtl.uint<2>) {}

  // Unable to infer domain of port, when port is driven by constant.

  // CHECK: firrtl.module @UnableToInferDomainOfPortDrivenByConstant() {
  // CHECK:   %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  // CHECK:   %foo_ClockDomain, %foo_i = firrtl.instance foo @Foo(in ClockDomain: !firrtl.domain of @ClockDomain, in i: !firrtl.uint<1> domains [ClockDomain])
  // CHECK:   %0 = firrtl.domain.anon : !firrtl.domain of @ClockDomain
  // CHECK:   firrtl.domain.define %foo_ClockDomain, %0
  // CHECK:   firrtl.matchingconnect %foo_i, %c0_ui1 : !firrtl.uint<1>
  // CHECK: }
  firrtl.module @UnableToInferDomainOfPortDrivenByConstant() {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %foo_i = firrtl.instance foo @Foo(in i: !firrtl.uint<1>)
    firrtl.matchingconnect %foo_i, %c0_ui1 : !firrtl.uint<1>
  }

  // Unable to infer domain of port, when port is driven by arithmetic on constant.

  // CHECK: firrtl.module @UnableToInferDomainOfPortDrivenByConstantExpr() {
  // CHECK:   %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  // CHECK:   %0 = firrtl.add %c0_ui1, %c0_ui1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
  // CHECK:   %bar_ClockDomain, %bar_i = firrtl.instance bar @Bar(in ClockDomain: !firrtl.domain of @ClockDomain, in i: !firrtl.uint<2> domains [ClockDomain])
  // CHECK:   %1 = firrtl.domain.anon : !firrtl.domain of @ClockDomain
  // CHECK:   firrtl.domain.define %bar_ClockDomain, %1
  // CHECK:   firrtl.matchingconnect %bar_i, %0 : !firrtl.uint<2>
  // CHECK: }
  firrtl.module @UnableToInferDomainOfPortDrivenByConstantExpr() {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %0 = firrtl.add %c0_ui1, %c0_ui1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
    %bar_i = firrtl.instance bar @Bar(in i: !firrtl.uint<2>)
    firrtl.matchingconnect %bar_i, %0 : !firrtl.uint<2>
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
