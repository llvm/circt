// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-infer-domains{mode=check}))' %s --verify-diagnostics

// in "check" mode, infer-domains will require that all ops are fully annotated
// with domains. No inference is run.

// CHECK-LABEL: MissingDomain
firrtl.circuit "MissingDomain" {
  firrtl.domain @ClockDomain

  // expected-note @below {{in module "MissingDomain"}}
  firrtl.module @MissingDomain(
    // expected-error @below {{missing "ClockDomain" association for port "x"}}
    in %x: !firrtl.uint<1>
  ) {}
}

// CHECK-LABEL: MissingSecondDomain
firrtl.circuit "MissingSecondDomain" {
  firrtl.domain @ClockDomain
  firrtl.domain @PowerDomain

  // expected-note @below {{in module "MissingSecondDomain"}}
  firrtl.module @MissingSecondDomain(
    in %c : !firrtl.domain of @ClockDomain,
    // expected-error @below {{missing "PowerDomain" association for port "x"}}
    in %x : !firrtl.uint<1> domains [%c]
  ) {}
}

// CHECK-LABEL: UndrivenOutputDomain
firrtl.circuit "UndrivenOutputDomain" {
  firrtl.domain @ClockDomain

  // expected-note @below {{in module "UndrivenOutputDomain"}}
  firrtl.module @UndrivenOutputDomain(
    // expected-error @below {{undriven domain port "c"}}
    out %c : !firrtl.domain of @ClockDomain
  ) {}
}

// CHECK-LABEL: UndrivenInstanceDomainPort
firrtl.circuit "UndrivenInstanceDomainPort" {
  firrtl.domain @ClockDomain

  firrtl.extmodule @Foo(in c : !firrtl.domain of @ClockDomain)

  firrtl.module @UndrivenInstanceDomainPort() {
    // expected-note  @+2 {{in instance "foo"}}
    // expected-error @+1 {{undriven domain port "c"}}
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

  firrtl.module @UndrivenInstanceChoiceDomainPort() {
    // expected-note  @+2 {{in instance_choice "inst"}} 
    // expected-error @+1 {{undriven domain port "c"}}
    %inst_c = firrtl.instance_choice inst @Foo alternatives @Option { @X -> @Bar } (in c : !firrtl.domain of @ClockDomain)
  }
}

// Test that domain crossing errors are still caught when in check-only mode.
// Catching this involves processing the module without writing back to the IR.
firrtl.circuit "IllegalDomainCrossing" {
  firrtl.domain @ClockDomain
  firrtl.module @IllegalDomainCrossing(
    in %A: !firrtl.domain of @ClockDomain,
    in %B: !firrtl.domain of @ClockDomain,
    // expected-note @below {{2nd operand has domains: [ClockDomain: A]}}
    in %a: !firrtl.uint<1> domains [%A],
    // expected-note @below {{1st operand has domains: [ClockDomain: B]}}
    out %b: !firrtl.uint<1> domains [%B]
  ) {
    // expected-error @below {{illegal domain crossing in operation}}
    firrtl.matchingconnect %b, %a : !firrtl.uint<1>
  }
}

// CHECK-LABEL: ExactDuplicateDomain
firrtl.circuit "DuplicateDomainEquivalence" {
  firrtl.domain @ClockDomain
  // expected-note @below {{in module "DuplicateDomainEquivalence"}}
  firrtl.module @DuplicateDomainEquivalence(
    // expected-note @below {{associated with "ClockDomain" port "A"}}
    in %A: !firrtl.domain of @ClockDomain,
    // expected-error @below {{duplicate "ClockDomain" association for port "a"}}
    in %a: !firrtl.uint<1> domains [%A, %A],
    out %b: !firrtl.uint<1> domains [%A]
  ) {
    firrtl.matchingconnect %b, %a : !firrtl.uint<1>
  }
}

// CHECK-LABEL: DuplicateDomainKind
firrtl.circuit "DuplicateDomainEquivalence" {
  firrtl.domain @ClockDomain
  // expected-note @below {{in module "DuplicateDomainEquivalence"}}
  firrtl.module @DuplicateDomainEquivalence(
    // expected-note @below {{associated with "ClockDomain" port "A"}}
    in  %A: !firrtl.domain of @ClockDomain,
    // expected-note @below {{associated with "ClockDomain" port "B"}}
    in  %B: !firrtl.domain of @ClockDomain,
    // expected-error @below {{duplicate "ClockDomain" association for port "a"}}
    in  %a: !firrtl.uint<1> domains [%A, %B],
    out %b: !firrtl.uint<1> domains [%A]
  ) {
    firrtl.matchingconnect %b, %a : !firrtl.uint<1>
  }
}
